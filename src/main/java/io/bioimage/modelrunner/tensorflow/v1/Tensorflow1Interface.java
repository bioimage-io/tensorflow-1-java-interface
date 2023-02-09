/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Tensorflow 1.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the BioImage.io nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */

package io.bioimage.modelrunner.tensorflow.v1;

import com.google.protobuf.InvalidProtocolBufferException;

import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensorflow.v1.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.tensorflow.v1.tensor.TensorBuilder;

import java.util.ArrayList;
import java.util.List;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

/**
 * This plugin includes the libraries to convert back and forth TensorFlow 1 to
 * Sequences and IcyBufferedImages.
 * 
 * @see ImgLib2Builder Create images from tensors.
 * @see TensorBuilder TensorBuilder: Create tensors from images and sequences.
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class Tensorflow1Interface implements DeepLearningEngineInterface {

	private static final String[] MODEL_TAGS = { "serve", "inference", "train",
		"eval", "gpu", "tpu" };

	private static final String[] TF_MODEL_TAGS = {
		"tf.saved_model.tag_constants.SERVING",
		"tf.saved_model.tag_constants.INFERENCE",
		"tf.saved_model.tag_constants.TRAINING",
		"tf.saved_model.tag_constants.EVAL", "tf.saved_model.tag_constants.GPU",
		"tf.saved_model.tag_constants.TPU" };

	private static final String[] SIGNATURE_CONSTANTS = { "serving_default",
		"inputs", "tensorflow/serving/classify", "classes", "scores", "inputs",
		"tensorflow/serving/predict", "outputs", "inputs",
		"tensorflow/serving/regress", "outputs", "train", "eval",
		"tensorflow/supervised/training", "tensorflow/supervised/eval" };

	private static final String[] TF_SIGNATURE_CONSTANTS = {
		"tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY",
		"tf.saved_model.signature_constants.CLASSIFY_INPUTS",
		"tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME",
		"tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES",
		"tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES",
		"tf.saved_model.signature_constants.PREDICT_INPUTS",
		"tf.saved_model.signature_constants.PREDICT_METHOD_NAME",
		"tf.saved_model.signature_constants.PREDICT_OUTPUTS",
		"tf.saved_model.signature_constants.REGRESS_INPUTS",
		"tf.saved_model.signature_constants.REGRESS_METHOD_NAME",
		"tf.saved_model.signature_constants.REGRESS_OUTPUTS",
		"tf.saved_model.signature_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY",
		"tf.saved_model.signature_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY",
		"tf.saved_model.signature_constants.SUPERVISED_TRAIN_METHOD_NAME",
		"tf.saved_model.signature_constants.SUPERVISED_EVAL_METHOD_NAME" };

	/**
	 * The loaded Tensorflow 1 model
	 */
	private static SavedModelBundle model;
	private static SignatureDef sig;

	public Tensorflow1Interface() {}

	@Override
	public void loadModel(String modelFolder, String modelSource)
		throws LoadModelException
	{
		model = SavedModelBundle.load(modelFolder, "serve");
		byte[] byteGraph = model.metaGraphDef();
		try {
			sig = MetaGraphDef.parseFrom(byteGraph).getSignatureDefOrThrow(
				"serving_default");
		}
		catch (InvalidProtocolBufferException e) {
			throw new LoadModelException();
		}
	}

	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors)
		throws RunModelException
	{
		Session session = model.session();
		Session.Runner runner = session.runner();
		List<String> inputListNames = new ArrayList<String>();
		List<org.tensorflow.Tensor<?>> inTensors =
			new ArrayList<org.tensorflow.Tensor<?>>();
		for (Tensor tt : inputTensors) {
			inputListNames.add(tt.getName());
			org.tensorflow.Tensor<?> inT = TensorBuilder.build(tt);
			inTensors.add(inT);
			runner.feed(getModelInputName(tt.getName()), inT);
		}

		for (Tensor tt : outputTensors)
			runner = runner.fetch(getModelOutputName(tt.getName()));
		// Run runner
		List<org.tensorflow.Tensor<?>> resultPatchTensors = runner.run();

		// Fill the agnostic output tensors list with data from the inference result
		outputTensors = fillOutputTensors(resultPatchTensors, outputTensors);
		for (org.tensorflow.Tensor<?> tt : inTensors) {
			tt.close();
		}
		for (org.tensorflow.Tensor<?> tt : resultPatchTensors) {
			tt.close();
		}
	}

	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by Deep Icy
	 * 
	 * @param outputNDArrays an NDList containing NDArrays (tensors)
	 * @param outputTensors the names given to the tensors by the model
	 * @return a list with Deep Learning framework agnostic tensors
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static List<Tensor<?>> fillOutputTensors(
		List<org.tensorflow.Tensor<?>> outputNDArrays,
		List<Tensor<?>> outputTensors) throws RunModelException
	{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i++) {
			outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
		}
		return outputTensors;
	}

	@Override
	public void closeModel() {
		sig = null;
		if (model != null) {
			model.session().close();
			model.close();
		}
		model = null;

	}

	// TODO make only one
	/**
	 * Retrieves the readable input name from the graph signature definition given
	 * the signature input name.
	 * 
	 * @param inputName Signature input name.
	 * @return The readable input name.
	 */
	public static String getModelInputName(String inputName) {
		TensorInfo inputInfo = sig.getInputsMap().getOrDefault(inputName, null);
		if (inputInfo != null) {
			String modelInputName = inputInfo.getName();
			if (modelInputName != null) {
				if (modelInputName.endsWith(":0")) {
					return modelInputName.substring(0, modelInputName.length() - 2);
				}
				else {
					return modelInputName;
				}
			}
			else {
				return inputName;
			}
		}
		return inputName;
	}

	/**
	 * Retrieves the readable output name from the graph signature definition
	 * given the signature output name.
	 * 
	 * @param outputName Signature output name.
	 * @return The readable output name.
	 */
	public static String getModelOutputName(String outputName) {
		TensorInfo outputInfo = sig.getOutputsMap().getOrDefault(outputName, null);
		if (outputInfo != null) {
			String modelOutputName = outputInfo.getName();
			if (modelOutputName.endsWith(":0")) {
				return modelOutputName.substring(0, modelOutputName.length() - 2);
			}
			else {
				return modelOutputName;
			}
		}
		else {
			return outputName;
		}
	}
}
