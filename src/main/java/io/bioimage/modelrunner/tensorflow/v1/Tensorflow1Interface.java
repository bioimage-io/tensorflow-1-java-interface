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
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensorflow.v1.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.tensorflow.v1.tensor.TensorBuilder;
import io.bioimage.modelrunner.tensorflow.v1.tensor.mappedbuffer.ImgLib2ToMappedBuffer;
import io.bioimage.modelrunner.tensorflow.v1.tensor.mappedbuffer.MappedBufferToImgLib2;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

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
     * Idetifier for the files that contain the data of the inputs
     */
    final private static String INPUT_FILE_TERMINATION = "_model_input";
    
    /**
     * Idetifier for the files that contain the data of the outputs
     */
    final private static String OUTPUT_FILE_TERMINATION = "_model_output";
    /**
     * Key for the inputs in the map that retrieves the file names for interprocess communication
     */
    final private static String INPUTS_MAP_KEY = "inputs";
    /**
     * Key for the outputs in the map that retrieves the file names for interprocess communication
     */
    final private static String OUTPUTS_MAP_KEY = "outputs";
    /**
     * File extension for the temporal files used for interprocessing
     */
    final private static String FILE_EXTENSION = ".dat";

    /**
     * The loaded Tensorflow 1 model
     */
	private static SavedModelBundle model;
	private static SignatureDef sig;
	
	private boolean interprocessing = false;
    
    private String tmpDir;
    
    private String modelFolder;
	
    public Tensorflow1Interface() throws IOException
    {
    	boolean isMac = PlatformDetection.isMacOS();
    	boolean isIntel = new PlatformDetection().getArch().equals(PlatformDetection.ARCH_X86_64);
    	if (isMac && isIntel) {
    		interprocessing = true;
    		tmpDir = getTemporaryDir();
    		
    	}
    }
	
    public Tensorflow1Interface(boolean doInterprocessing) throws IOException
    {
    	if (!doInterprocessing) {
    		interprocessing = false;
    	} else {
    		boolean isMac = PlatformDetection.isMacOS();
        	boolean isIntel = new PlatformDetection().getArch().equals(PlatformDetection.ARCH_X86_64);
        	if (isMac && isIntel) {
        		interprocessing = true;
        		tmpDir = getTemporaryDir();
        		
        	}
    	}
    }

	@Override
	public void loadModel(String modelFolder, String modelSource)
		throws LoadModelException
	{
		if (interprocessing) {
			this.modelFolder = modelFolder;
			return;
		}
		model = SavedModelBundle.load(modelFolder, "serve");
		byte[] byteGraph = model.metaGraphDef();
		try {
			sig = MetaGraphDef.parseFrom(byteGraph).getSignatureDefOrThrow(
				"serving_default");
		}
		catch (InvalidProtocolBufferException e) {
			closeModel();
			throw new LoadModelException();
		}
	}

	@Override
	public void run(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors)
		throws RunModelException
	{
		if (interprocessing) {
			runInterprocessing(inputTensors, outputTensors);
			return;
		}
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
	
	public void runInterprocessing(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		createTensorsForInterprocessing(inputTensors);
		createTensorsForInterprocessing(outputTensors);
		List<String> args = getProcessCommandsWithoutArgs();
		args.add(modelFolder);
		args.add(this.tmpDir);
		for (Tensor tensor : inputTensors) {args.add(tensor.getName() + INPUT_FILE_TERMINATION);}
		for (Tensor tensor : outputTensors) {args.add(tensor.getName() + OUTPUT_FILE_TERMINATION);}
		ProcessBuilder builder = new ProcessBuilder(args);
        Process process;
		try {
			process = builder.inheritIO().start();
	        if (process.waitFor() != 0)
	        	throw new RunModelException("Error executing the Tensorflow 1 model in"
	        			+ " a separate process. The process was not terminated correctly.");
		} catch (RunModelException e) {
			closeModel();
			throw e;
		} catch (Exception e) {
			closeModel();
			throw new RunModelException(e.getCause().toString());
		}
		
		retrieveInterprocessingTensors(outputTensors);
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
	
	
	/**
	 * Methods to run interprocessing and bypass the errors that occur in MacOS intel
	 * with the compatibility between TF1 and TF2
	 */
    
    public static void main(String[] args) throws LoadModelException, IOException, RunModelException {
    	// Unpack the args needed
    	if (args.length < 4)
    		throw new IllegalArgumentException("Error exectuting Tensorflow 1, "
    				+ "at least 5 arguments are required:" + System.lineSeparator()
    				+ " - Folder where the model is located" + System.lineSeparator()
    				+ " - Temporary dir where the memory mapped files are located" + System.lineSeparator()
    				+ " - Name of the model input followed by the String + '_model_input'" + System.lineSeparator()
    				+ " - Name of the second model input (if it exists) followed by the String + '_model_input'" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Name of the nth model input (if it exists)  followed by the String + '_model_input'" + System.lineSeparator()
    				+ " - Name of the model output followed by the String + '_model_output'" + System.lineSeparator()
    				+ " - Name of the second model output (if it exists) followed by the String + '_model_output'" + System.lineSeparator()
    				+ " - ...." + System.lineSeparator()
    				+ " - Name of the nth model output (if it exists)  followed by the String + '_model_output'" + System.lineSeparator()
    				);
    	String modelFolder = args[0];
    	if (!(new File(modelFolder).isDirectory())) {
    		throw new IllegalArgumentException("Argument 0 of the main method, '" + modelFolder + "' "
    				+ "should be an existing directory containing a Tensorflow 1 model.");
    	}
    	
    	Tensorflow1Interface tfInterface = new Tensorflow1Interface(false);
    	tfInterface.tmpDir = args[1];
    	if (!(new File(args[1]).isDirectory())) {
    		throw new IllegalArgumentException("Argument 1 of the main method, '" + args[1] + "' "
    				+ "should be an existing directory.");
    	}
    	
    	tfInterface.loadModel(modelFolder, modelFolder);
    	
    	HashMap<String, List<String>> map = getInputTensorsFileNames(args);
    	List<String> inputNames = map.get(INPUTS_MAP_KEY);
    	List<Tensor<?>> inputList = inputNames.stream().map(n -> {
									try {
										return tfInterface.retrieveInterprocessingTensorsByName(n);
									} catch (RunModelException e) {
										return null;
									}
								}).collect(Collectors.toList());
    	List<String> outputNames = map.get(OUTPUTS_MAP_KEY);
    	List<Tensor<?>> outputList = outputNames.stream().map(n -> {
									try {
										return tfInterface.retrieveInterprocessingTensorsByName(n);
									} catch (RunModelException e) {
										return null;
									}
								}).collect(Collectors.toList());
    	tfInterface.run(inputList, outputList);
    	tfInterface.createTensorsForInterprocessing(outputList);
    }
	
	private void createTensorsForInterprocessing(List<Tensor<?>> tensors) throws RunModelException{
		for (Tensor<?> tensor : tensors) {
			long lenFile = ImgLib2ToMappedBuffer.findTotalLengthFile(tensor);
			try (RandomAccessFile rd = 
    				new RandomAccessFile(tmpDir + File.separator + tensor.getName() + FILE_EXTENSION, "rw");
    				FileChannel fc = rd.getChannel();) {
    			MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_WRITE, 0, lenFile);
    			ByteBuffer byteBuffer = mem.duplicate();
    			byteBuffer.put(ImgLib2ToMappedBuffer.createFileHeader(tensor));
    			ImgLib2ToMappedBuffer.build(tensor, byteBuffer);
    		} catch (IOException e) {
    			closeModel();
    			throw new RunModelException(e.getCause().toString());
			}
		}
	}
	
	private void retrieveInterprocessingTensors(List<Tensor<?>> tensors) throws RunModelException{
		for (Tensor<?> tensor : tensors) {
			try (RandomAccessFile rd = 
    				new RandomAccessFile(tmpDir + File.separator + tensor.getName() + FILE_EXTENSION, "r");
    				FileChannel fc = rd.getChannel();) {
    			MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
    			ByteBuffer byteBuffer = mem.duplicate();
    			tensor.setData(MappedBufferToImgLib2.build(byteBuffer));
    		} catch (IOException e) {
    			closeModel();
    			throw new RunModelException(e.getCause().toString());
			}
		}
	}
	
	private < T extends RealType< T > & NativeType< T > > Tensor<T> 
				retrieveInterprocessingTensorsByName(String name) throws RunModelException {
		try (RandomAccessFile rd = 
				new RandomAccessFile(tmpDir + File.separator + name + FILE_EXTENSION, "r");
				FileChannel fc = rd.getChannel();) {
			MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
			ByteBuffer byteBuffer = mem.duplicate();
			return MappedBufferToImgLib2.buildTensor(byteBuffer);
		} catch (IOException e) {
			closeModel();
			throw new RunModelException(e.getCause().toString());
		}
	}
	
	/**
	 * Create the arguments needed to execute tensorflow1 in another 
	 * process with the corresponding tensors
	 * @return
	 */
	private List<String> getProcessCommandsWithoutArgs() {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +  File.separator + "bin" + File.separator + "java";
        String classpath = System.getProperty("java.class.path");
        ProtectionDomain protectionDomain = Tensorflow1Interface.class.getProtectionDomain();
        CodeSource codeSource = protectionDomain.getCodeSource();
        String className = Tensorflow1Interface.class.getName();
        classpath += File.pathSeparator;
        for (File ff : new File(codeSource.getLocation().getPath()).getParentFile().listFiles()) {
        	classpath += ff.getAbsolutePath() + File.pathSeparator;
        }
        List<String> command = new LinkedList<String>();
        command.add(javaBin);
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        return command;
	}
	
	/**
	 * Get temporary directory to perform the interprocessing communication in MacOSX intel
	 * @return the tmp dir
	 * @throws IOException
	 */
	private static String getTemporaryDir() throws IOException {
		String tmpDir;
		if (System.getenv("temp") != null
			&& Files.isWritable(Paths.get(System.getenv("temp")))) {
			return System.getenv("temp");
		} else if (System.getenv("TEMP") != null
			&& Files.isWritable(Paths.get(System.getenv("TEMP")))) {
			return System.getenv("TEMP");
		} else if (System.getenv("tmp") != null
			&& Files.isWritable(Paths.get(System.getenv("tmp")))) {
			return System.getenv("tmp");
		} else if (System.getenv("TMP") != null
			&& Files.isWritable(Paths.get(System.getenv("TMP")))) {
			return System.getenv("TMP");
		} else if (System.getProperty("java.io.tmpdir") != null 
				&& Files.isWritable(Paths.get(System.getProperty("java.io.tmpdir")))) {
			return System.getProperty("java.io.tmpdir");
		}
		String enginesDir = getEnginesDir();
		if (Files.isWritable(Paths.get(enginesDir))) {
			tmpDir = enginesDir + File.separator + "temp";
			if (!(new File(tmpDir).isDirectory()) &&  !(new File(tmpDir).mkdirs()))
				tmpDir = enginesDir;
		} else {
			throw new IOException("Unable to find temporal directory with writting rights. "
					+ "Please either allow writting on the system temporal folder or on '" + enginesDir + "'.");
		}
		return tmpDir;
	}
	
	private static String getEnginesDir() {
		ProtectionDomain protectionDomain = Tensorflow1Interface.class.getProtectionDomain();
        CodeSource codeSource = protectionDomain.getCodeSource();
        String jarFile = codeSource.getLocation().getPath();
        return jarFile;
	}
    
    /**
     * Retrieve the file names used for interprocess communication
     * @param args
     * 	args provided to the main method
     * @return a map with a list of input and output names
     */
    private static HashMap<String, List<String>> getInputTensorsFileNames(String[] args) {
    	List<String> inputNames = new ArrayList<String>();
    	List<String> outputNames = new ArrayList<String>();
    	for (int i = 2; i < args.length; i ++) {
    		if (args[i].endsWith(INPUT_FILE_TERMINATION))
    			inputNames.add(args[i].substring(0, args[i].length() - INPUT_FILE_TERMINATION.length()));
    		else if (args[i].endsWith(OUTPUT_FILE_TERMINATION))
    			outputNames.add(args[i].substring(0, args[i].length() - OUTPUT_FILE_TERMINATION.length()));
    	}
    	if (inputNames.size() == 0)
    		throw new IllegalArgumentException("The args to the main method of '" 
    						+ Tensorflow1Interface.class.toString() + "' should contain at "
    						+ "least one input, defined as '<input_name> + '" + INPUT_FILE_TERMINATION + "'.");
    	if (outputNames.size() == 0)
    		throw new IllegalArgumentException("The args to the main method of '" 
					+ Tensorflow1Interface.class.toString() + "' should contain at "
					+ "least one output, defined as '<output_name> + '" + OUTPUT_FILE_TERMINATION + "'.");
    	HashMap<String, List<String>> map = new HashMap<String, List<String>>();
    	map.put(INPUTS_MAP_KEY, inputNames);
    	map.put(OUTPUTS_MAP_KEY, outputNames);
    	return map;
    }
}
