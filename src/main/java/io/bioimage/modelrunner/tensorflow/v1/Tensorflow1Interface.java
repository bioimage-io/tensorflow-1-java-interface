/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Tensorflow 1.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.tensorflow.v1;

import com.google.gson.Gson;
import com.google.protobuf.InvalidProtocolBufferException;

import io.bioimage.modelrunner.apposed.appose.Service;
import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.apposed.appose.Service.Task;
import io.bioimage.modelrunner.apposed.appose.Service.TaskStatus;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.description.ModelDescriptorFactory;
import io.bioimage.modelrunner.bioimageio.download.DownloadModel;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.tensorflow.v1.shm.ShmBuilder;
import io.bioimage.modelrunner.tensorflow.v1.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.tensorflow.v1.tensor.TensorBuilder;
import io.bioimage.modelrunner.utils.CommonUtils;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.ZipUtils;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;

import java.io.File;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.yaml.snakeyaml.Yaml;

/**
 * TODO 
 * currently this class does not need interprocessing as the conflicting 
 * engine (TF2) is hte one that is going to do the interprocessing. However, 
 * the code is still structured for interprocessing as I assume in the near
 * future I am going to start moving everything towards interprocessing, although
 * it wil probably involve more omplex architectures than mappedbuffers
 * TODO
 * 
 * Class to that communicates with the dl-model runner, see 
 * @see <a href="https://github.com/bioimage-io/model-runner-java">dlmodelrunner</a>
 * to execute Tensorflow 1 models.
 * This class implements the interface {@link DeepLearningEngineInterface} to get the 
 * agnostic {@link io.bioimage.modelrunner.tensor.Tensor}, convert them into 
 * {@link org.tensorflow.Tensor}, execute a Tensorflow 1 Deep Learning model on them and
 * convert the results back to {@link io.bioimage.modelrunner.tensor.Tensor} to send them 
 * to the main program in an agnostic manner.
 * 
 * {@link ImgLib2Builder}. Creates ImgLib2 images for the backend
 *  of {@link io.bioimage.modelrunner.tensor.Tensor} from {@link org.tensorflow.Tensor}
 * {@link TensorBuilder}. Converts {@link io.bioimage.modelrunner.tensor.Tensor} into {@link org.tensorflow.Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public class Tensorflow1Interface implements DeepLearningEngineInterface {
	/**
	 * Name without vesion of the JAR created for this library
	 */
	private static final String JAR_FILE_NAME = "dl-modelrunner-tensorflow-";

	private static final String NAME_KEY = "name";
	private static final String SHAPE_KEY = "shape";
	private static final String DTYPE_KEY = "dtype";
	private static final String IS_INPUT_KEY = "isInput";
	private static final String MEM_NAME_KEY = "memoryName";
	
	private List<SharedMemoryArray> shmaInputList = new ArrayList<SharedMemoryArray>();
	
	private List<SharedMemoryArray> shmaOutputList = new ArrayList<SharedMemoryArray>();
	
	private List<String> shmaNamesList = new ArrayList<String>();

    /**
     * The loaded Tensorflow 2 model
     */
	private static SavedModelBundle model;
	/**
	 * Internal object of the Tensorflow model
	 */
	private static SignatureDef sig;
	/**
	 * Whether the execution needs interprocessing (MacOS Interl) or not
	 */
	private boolean interprocessing = false;
    /**
     * Folde containing the model that is being executed
     */
    private String modelFolder;
    /**
     * Process where the model is being loaded and executed
     */
    Service runner;
    
    /**
     * TODO the interprocessing is executed for every OS
     * Constructor that detects whether the operating system where it is being 
     * executed is Windows or Mac or not to know if it is going to need interprocessing 
     * or not
     * @throws IOException if the temporary dir is not found
     * @throws URISyntaxException 
     */
    public Tensorflow1Interface() throws IOException, URISyntaxException
    {
    	this(true);
    }
	
    /**
     * Private constructor that can only be launched from the class to create a separate
     * process to avoid the conflicts that occur in the same process between TF2 and TF1/Pytorch
     * in Windows and Mac
     * @param doInterprocessing
     * 	whether to do interprocessing or not
     * @throws IOException if the temp dir is not found
     * @throws URISyntaxException 
     */
    protected Tensorflow1Interface(boolean doInterprocessing) throws IOException, URISyntaxException
    {
		interprocessing = doInterprocessing;
		if (this.interprocessing) {
			runner = getRunner();
			runner.debug((text) -> System.err.println(text));
		}
    }
    
    private Service getRunner() throws IOException, URISyntaxException {
		List<String> args = getProcessCommandsWithoutArgs();
		String[] argArr = new String[args.size()];
		args.toArray(argArr);

		return new Service(new File("."), argArr);
    }

    /**
     * {@inheritDoc}
     * 
     * Load a Tensorflow 1 model. If the machine where the code is
     * being executed is a MacOS Intel, the model will be loaded in 
     * a separate process each time the method {@link #run(List, List)}
     * is called 
     */
	@Override
	public void loadModel(String modelFolder, String modelSource)
		throws LoadModelException
	{
		this.modelFolder = modelFolder;
		if (interprocessing) {
			try {
				launchModelLoadOnProcess();
			} catch (IOException | InterruptedException e) {
				throw new LoadModelException(Types.stackTrace(e));
			}
			return;
		}
		try {
			checkModelUnzipped();
		} catch (Exception e) {
			throw new LoadModelException(e.toString());
		}
		model = SavedModelBundle.load(this.modelFolder, "serve");
		byte[] byteGraph = model.metaGraphDef();
		try {
			sig = MetaGraphDef.parseFrom(byteGraph).getSignatureDefOrThrow(
				"serving_default");
		}
		catch (InvalidProtocolBufferException e) {
			closeModel();
			throw new LoadModelException(Types.stackTrace(e));
		}
	}
	
	private void launchModelLoadOnProcess() throws IOException, InterruptedException {
		HashMap<String, Object> args = new HashMap<String, Object>();
		args.put("modelFolder", modelFolder);
		Task task = runner.task("loadModel", args);
		task.waitFor();
		if (task.status == TaskStatus.CANCELED)
			throw new RuntimeException();
		else if (task.status == TaskStatus.FAILED)
			throw new RuntimeException(task.error);
		else if (task.status == TaskStatus.CRASHED) {
			this.runner.close();
			runner = null;
			throw new RuntimeException(task.error);
		}
	}
	
	/**
	 * Check if an unzipped tensorflow model exists in the model folder, 
	 * and if not look for it and unzip it
	 * @throws LoadModelException if no model is found
	 * @throws IOException if there is any error unzipping the model
	 * @throws Exception if there is any error related to model packaging
	 */
	private void checkModelUnzipped() throws LoadModelException, IOException, Exception {
		if (new File(modelFolder, "variables").isDirectory()
				&& new File(modelFolder, "saved_model.pb").isFile())
			return;
		unzipTfWeights(ModelDescriptorFactory.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME));
	}
	
	/**
	 * Method that unzips the tensorflow model zip into the variables
	 * folder and .pb file, if they are saved in a zip
	 * @throws LoadModelException if not zip file is found
	 * @throws IOException if there is any error unzipping
	 */
	private void unzipTfWeights(ModelDescriptor descriptor) throws LoadModelException, IOException {
		if (new File(modelFolder, "tf_weights.zip").isFile()) {
			System.out.println("Unzipping model...");
			ZipUtils.unzipFolder(modelFolder + File.separator + "tf_weights.zip", modelFolder);
		} else if ( descriptor.getWeights().getAllSuportedWeightNames()
				.contains(EngineInfo.getBioimageioTfKey()) ) {
			String source = descriptor.getWeights().gettAllSupportedWeightObjects().stream()
					.filter(ww -> ww.getFramework().equals(EngineInfo.getBioimageioTfKey()))
					.findFirst().get().getSource();
			if (new File(source).isFile()) {
				System.out.println("Unzipping model...");
				ZipUtils.unzipFolder(new File(source).getAbsolutePath(), modelFolder);
			} else if (new File(modelFolder, source).isFile()) {
				System.out.println("Unzipping model...");
				ZipUtils.unzipFolder(new File(modelFolder, source).getAbsolutePath(), modelFolder);
			} else {
				source = DownloadModel.getFileNameFromURLString(source);
				System.out.println("Unzipping model...");
				ZipUtils.unzipFolder(modelFolder + File.separator + source, modelFolder);
			}
		} else {
			throw new LoadModelException("No model file was found in the model folder");
		}
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Run a Tensorflow1 model on the data provided by the {@link Tensor} input list
	 * and modifies the output list with the results obtained
	 */
	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void run(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors)
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
		int c = 0;
		for (Tensor<T> tt : inputTensors) {
			inputListNames.add(tt.getName());
			org.tensorflow.Tensor<?> inT = TensorBuilder.build(tt);
			inTensors.add(inT);
			String inputName = getModelInputName(tt.getName(), c ++);
			runner.feed(inputName, inT);
		}
		c = 0;
		for (Tensor<R> tt : outputTensors)
			runner = runner.fetch(getModelOutputName(tt.getName(), c ++));
		// Run runner
		List<org.tensorflow.Tensor<?>> resultPatchTensors = runner.run();

		// Fill the agnostic output tensors list with data from the inference result
		fillOutputTensors(resultPatchTensors, outputTensors);
		for (org.tensorflow.Tensor<?> tt : inTensors) {
			tt.close();
		}
		for (org.tensorflow.Tensor<?> tt : resultPatchTensors) {
			tt.close();
		}
	}

	@Override
	public <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>> List<RandomAccessibleInterval<R>> inference(
			List<RandomAccessibleInterval<T>> inputs) throws RunModelException {
		if (interprocessing) {
			return runInterprocessing(inputs);
		}
		Session session = model.session();
		Session.Runner runner = session.runner();
		List<org.tensorflow.Tensor<?>> inTensors =
			new ArrayList<org.tensorflow.Tensor<?>>();
		int c = 0;
		for (RandomAccessibleInterval<T> tt : inputs) {
			org.tensorflow.Tensor<?> inT = TensorBuilder.build(tt);
			inTensors.add(inT);
			String inputName = getModelInputName("input" + c, c ++);
			runner.feed(inputName, inT);
		}
		c = 0;
		List<String> outputInfo = sig.getOutputsMap().values().stream()
				.map(nn -> {
					String name = nn.getName();
					if (name.endsWith(":0"))
						return name.substring(0, name.length() - 2);
					return name;
				}).collect(Collectors.toList());

		for (String name : outputInfo)
			runner = runner.fetch(name);
		// Run runner
		List<org.tensorflow.Tensor<?>> resultPatchTensors = runner.run();
		for (org.tensorflow.Tensor<?> tt : inTensors)
			tt.close();
		List<RandomAccessibleInterval<R>> rais = new ArrayList<RandomAccessibleInterval<R>>();
		for (int i = 0; i < resultPatchTensors.size(); i++) {
			try {
				rais.add(ImgLib2Builder.build(resultPatchTensors.get(i)));
				resultPatchTensors.get(i).close();
			} catch (IllegalArgumentException ex) {
				for (org.tensorflow.Tensor<?> tt : resultPatchTensors)
					tt.close();
				throw new RunModelException(Types.stackTrace(ex));
			}
		}
		return rais;
	}
	
	protected void runFromShmas(List<String> inputs, List<String> outputs) throws IOException {
		Session session = model.session();
		Session.Runner runner = session.runner();
		
		List<org.tensorflow.Tensor<?>> inTensors = new ArrayList<org.tensorflow.Tensor<?>>();
		int c = 0;
		for (String ee : inputs) {
			Map<String, Object> decoded = Types.decode(ee);
			SharedMemoryArray shma = SharedMemoryArray.read((String) decoded.get(MEM_NAME_KEY));
			org.tensorflow.Tensor<?> inT = io.bioimage.modelrunner.tensorflow.v1.shm.TensorBuilder.build(shma);
			if (PlatformDetection.isWindows()) shma.close();
			inTensors.add(inT);
			String inputName = getModelInputName((String) decoded.get(NAME_KEY), c ++);
			runner.feed(inputName, inT);
		}
		
		c = 0;
		for (String ee : outputs)
			runner = runner.fetch(getModelOutputName((String) Types.decode(ee).get(NAME_KEY), c ++));
		// Run runner
		List<org.tensorflow.Tensor<?>> resultPatchTensors = runner.run();

		// Fill the agnostic output tensors list with data from the inference result
		c = 0;
		for (String ee : outputs) {
			Map<String, Object> decoded = Types.decode(ee);
			ShmBuilder.build((org.tensorflow.Tensor<?>) resultPatchTensors.get(c ++), (String) decoded.get(MEM_NAME_KEY));
		}
		// Close the remaining resources
		for (org.tensorflow.Tensor<?> tt : inTensors) {
			tt.close();
		}
		for (org.tensorflow.Tensor<?> tt : resultPatchTensors) {
			tt.close();
		}
	}
	
	protected List<String> inferenceFromShmas(List<String> inputs) throws IOException, RunModelException {
		Session session = model.session();
		Session.Runner runner = session.runner();
		List<org.tensorflow.Tensor<?>> inTensors =
			new ArrayList<org.tensorflow.Tensor<?>>();
		int c = 0;
		for (String ee : inputs) {
			Map<String, Object> decoded = Types.decode(ee);
			SharedMemoryArray shma = SharedMemoryArray.read((String) decoded.get(MEM_NAME_KEY));
			org.tensorflow.Tensor<?> inT = io.bioimage.modelrunner.tensorflow.v1.shm.TensorBuilder.build(shma);
			if (PlatformDetection.isWindows()) shma.close();
			inTensors.add(inT);
			String inputName = getModelInputName((String) decoded.get(NAME_KEY), c ++);
			runner.feed(inputName, inT);
		}
		List<String> outputInfo = sig.getOutputsMap().values().stream()
				.map(nn -> {
					String name = nn.getName();
					if (name.endsWith(":0"))
						return name.substring(0, name.length() - 2);
					return name;
				}).collect(Collectors.toList());

		for (String name : outputInfo)
			runner = runner.fetch(name);
		// Run runner
		List<org.tensorflow.Tensor<?>> resultPatchTensors = runner.run();
		for (org.tensorflow.Tensor<?> tt : inTensors)
			tt.close();

		shmaNamesList = new ArrayList<String>();
		for (int i = 0; i < resultPatchTensors.size(); i ++) {
			String name = SharedMemoryArray.createShmName();
			ShmBuilder.build(resultPatchTensors.get(i), name, false);
			shmaNamesList.add(name);
		}
		for (org.tensorflow.Tensor<?> tt : resultPatchTensors) {
			tt.close();
		}
		return shmaNamesList;
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	void runInterprocessing(List<Tensor<T>> inputTensors, List<Tensor<R>> outputTensors) throws RunModelException {
		shmaInputList = new ArrayList<SharedMemoryArray>();
		shmaOutputList = new ArrayList<SharedMemoryArray>();
		List<String> encIns = encodeInputs(inputTensors);
		List<String> encOuts = encodeOutputs(outputTensors);
		LinkedHashMap<String, Object> args = new LinkedHashMap<String, Object>();
		args.put("inputs", encIns);
		args.put("outputs", encOuts);

		try {
			Task task = runner.task("run", args);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException(task.error);
			}
			for (int i = 0; i < outputTensors.size(); i ++) {
	        	String name = (String) Types.decode(encOuts.get(i)).get(MEM_NAME_KEY);
	        	SharedMemoryArray shm = shmaOutputList.stream()
	        			.filter(ss -> ss.getName().equals(name)).findFirst().orElse(null);
	        	if (shm == null) {
	        		shm = SharedMemoryArray.read(name);
	        		shmaOutputList.add(shm);
	        	}
	        	RandomAccessibleInterval<T> rai = shm.getSharedRAI();
	        	outputTensors.get(i).setData(Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(rai), Util.getTypeFromInterval(Cast.unchecked(rai))));
	        }
		} catch (Exception e) {
			closeShmas();
			if (e instanceof RunModelException)
				throw (RunModelException) e;
			throw new RunModelException(Types.stackTrace(e));
		}
		closeShmas();
	}
	
	private <T extends RealType<T> & NativeType<T>, R extends RealType<R> & NativeType<R>>
	List<RandomAccessibleInterval<R>> runInterprocessing(List<RandomAccessibleInterval<T>> inputs) throws RunModelException {
		shmaInputList = new ArrayList<SharedMemoryArray>();
		List<String> encIns = new ArrayList<String>();
		Gson gson = new Gson();
		for (RandomAccessibleInterval<T> tt : inputs) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt, false, true);
			shmaInputList.add(shma);
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(SHAPE_KEY, tt.dimensionsAsLongArray());
			map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt));
			map.put(IS_INPUT_KEY, true);
			map.put(MEM_NAME_KEY, shma.getName());
			encIns.add(gson.toJson(map));
		}
		LinkedHashMap<String, Object> args = new LinkedHashMap<String, Object>();
		args.put("inputs", encIns);

		try {
			Task task = runner.task("inference", args);
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException(task.error);
			} else if (task.outputs == null)
				throw new RuntimeException("No outputs generated");
			List<String> outputs = (List<String>) task.outputs.get("encoded");
			List<RandomAccessibleInterval<R>> rais = new ArrayList<RandomAccessibleInterval<R>>();
			for (String out : outputs) {
	        	String name = (String) Types.decode(out).get(MEM_NAME_KEY);
	        	SharedMemoryArray shm = SharedMemoryArray.read(name);
	        	RandomAccessibleInterval<R> rai = shm.getSharedRAI();
	        	rais.add(Tensor.createCopyOfRaiInWantedDataType(Cast.unchecked(rai), Util.getTypeFromInterval(Cast.unchecked(rai))));
	        	shm.close();
			}
			closeShmas();
			return rais;
		} catch (Exception e) {
			closeShmas();
			if (e instanceof RunModelException)
				throw (RunModelException) e;
			throw new RunModelException(Types.stackTrace(e));
		}
	}
	
	private void closeInterprocess() throws RunModelException {
		try {
			Task task = runner.task("closeTensors");
			task.waitFor();
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException(task.error);
			}
		} catch (Exception e) {
			if (e instanceof RunModelException)
				throw (RunModelException) e;
			throw new RunModelException(Types.stackTrace(e));
		}
	}
	
	protected void closeFromInterp() {
		if (!PlatformDetection.isWindows())
			return;
		this.shmaNamesList.stream().forEach(nn -> {
			try {
				SharedMemoryArray.read(nn).close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		});
	}
	
	private void closeShmas() throws RunModelException {
		shmaInputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaInputList = null;
		shmaOutputList.forEach(shm -> {
			try { shm.close(); } catch (IOException e1) { e1.printStackTrace();}
		});
		shmaOutputList = null;
		if (interprocessing)
			closeInterprocess();
	}
	
	
	private <T extends RealType<T> & NativeType<T>> List<String> encodeInputs(List<Tensor<T>> inputTensors) {
		List<String> encodedInputTensors = new ArrayList<String>();
		Gson gson = new Gson();
		for (Tensor<T> tt : inputTensors) {
			SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
			shmaInputList.add(shma);
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(SHAPE_KEY, tt.getShape());
			map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
			map.put(IS_INPUT_KEY, true);
			map.put(MEM_NAME_KEY, shma.getName());
			encodedInputTensors.add(gson.toJson(map));
		}
		return encodedInputTensors;
	}
	
	
	private <T extends RealType<T> & NativeType<T>> 
	List<String> encodeOutputs(List<Tensor<T>> outputTensors) {
		Gson gson = new Gson();
		List<String> encodedOutputTensors = new ArrayList<String>();
		for (Tensor<?> tt : outputTensors) {
			HashMap<String, Object> map = new HashMap<String, Object>();
			map.put(NAME_KEY, tt.getName());
			map.put(IS_INPUT_KEY, false);
			if (!tt.isEmpty()) {
				map.put(SHAPE_KEY, tt.getShape());
				map.put(DTYPE_KEY, CommonUtils.getDataTypeFromRAI(tt.getData()));
				SharedMemoryArray shma = SharedMemoryArray.createSHMAFromRAI(tt.getData(), false, true);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else if (PlatformDetection.isWindows()){
				SharedMemoryArray shma = SharedMemoryArray.create(0);
				shmaOutputList.add(shma);
				map.put(MEM_NAME_KEY, shma.getName());
			} else {
				String memName = SharedMemoryArray.createShmName();
				map.put(MEM_NAME_KEY, memName);
				shmaNamesList.add(memName);
			}
			encodedOutputTensors.add(gson.toJson(map));
		}
		return encodedOutputTensors;
	}

	/**
	 * Create the list a list of output tensors agnostic to the Deep Learning
	 * engine that can be readable by Deep Icy
	 * 
	 * @param outputNDArrays an NDList containing NDArrays (tensors)
	 * @param outputTensors the names given to the tensors by the model
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static <T extends RealType<T> & NativeType<T>>  void fillOutputTensors(
												List<org.tensorflow.Tensor<?>> outputNDArrays,
												List<Tensor<T>> outputTensors) throws RunModelException
	{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i++) {
			try {
				outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
			} catch (IllegalArgumentException ex) {
				throw new RunModelException(Types.stackTrace(ex));
			}
		}
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Close the Tensorflow 1 {@link #model} and {@link #sig}. For 
	 * MacOS Intel systems it aso deletes the temporary files created to
	 * communicate with the other process
	 */
	@Override
	public void closeModel() {
		if (this.interprocessing && runner != null) {
			Task task;
			try {
				task = runner.task("close");
				task.waitFor();
			} catch (IOException | InterruptedException e) {
				throw new RuntimeException(Types.stackTrace(e));
			}
			if (task.status == TaskStatus.CANCELED)
				throw new RuntimeException();
			else if (task.status == TaskStatus.FAILED)
				throw new RuntimeException(task.error);
			else if (task.status == TaskStatus.CRASHED) {
				this.runner.close();
				runner = null;
				throw new RuntimeException(task.error);
			}
			this.runner.close();
			this.runner = null;
			return;
		} else if (this.interprocessing) {
			return;
		}
		if (model != null) {
			model.session().close();
			model.close();
		}
		model = null;
		sig = null;
	}

	// TODO make only one
	/**
	 * Retrieves the readable input name from the graph signature definition given
	 * the signature input name.
	 * 
	 * @param inputName Signature input name.
	 * @param i position of the input of interest in the list of inputs
	 * @return The readable input name.
	 */
	public static String getModelInputName(String inputName, int i) {
		TensorInfo inputInfo = sig.getInputsMap().getOrDefault(inputName, null);
		if (inputInfo == null) {
			inputInfo = sig.getInputsMap().values().stream().collect(Collectors.toList()).get(i);
		}
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
	 * @param i position of the input of interest in the list of inputs
	 * @return The readable output name.
	 */
	public static String getModelOutputName(String outputName, int i) {
		TensorInfo outputInfo = sig.getOutputsMap().getOrDefault(outputName, null);
		if (outputInfo == null) {
			outputInfo = sig.getOutputsMap().values().stream().collect(Collectors.toList()).get(i);
		}
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
	 * if java bin dir contains any special char, surround it by double quotes
	 * @param javaBin
	 * 	java bin dir
	 * @return impored java bin dir if needed
	 */
	private static String padSpecialJavaBin(String javaBin) {
		String[] specialChars = new String[] {" "};
        for (String schar : specialChars) {
        	if (javaBin.contains(schar) && PlatformDetection.isWindows()) {
        		return "\"" + javaBin + "\"";
        	}
        }
        return javaBin;
	}
	
	/**
	 * Create the arguments needed to execute tensorflow 2 in another 
	 * process with the corresponding tensors
	 * @return the command used to call the separate process
	 * @throws IOException if the command needed to execute interprocessing is too long
	 * @throws URISyntaxException if there is any error with the URIs retrieved from the classes
	 */
	private List<String> getProcessCommandsWithoutArgs() throws IOException, URISyntaxException {
		String javaHome = System.getProperty("java.home");
        String javaBin = javaHome +  File.separator + "bin" + File.separator + "java";

        String classpath = getCurrentClasspath();
        ProtectionDomain protectionDomain = Tensorflow1Interface.class.getProtectionDomain();
        String codeSource = protectionDomain.getCodeSource().getLocation().getPath();
        String f_name = URLDecoder.decode(codeSource, StandardCharsets.UTF_8.toString());
        f_name = new File(f_name).getAbsolutePath();
        for (File ff : new File(f_name).getParentFile().listFiles()) {
        	if (ff.getName().startsWith(JAR_FILE_NAME) && !ff.getAbsolutePath().equals(f_name))
        		continue;
        	classpath += ff.getAbsolutePath() + File.pathSeparator;
        }
        String className = JavaWorker.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(padSpecialJavaBin(javaBin));
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        return command;
	}
	
    private static String getCurrentClasspath() throws UnsupportedEncodingException {

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        String gsonPath = getPathFromClass(Gson.class);
        String snakeYaml = getPathFromClass(Yaml.class);
        String jnaPath = getPathFromClass(com.sun.jna.Library.class);
        String jnaPlatformPath = getPathFromClass(com.sun.jna.platform.FileUtils.class);
        if (modelrunnerPath == null || (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator)))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        classpath =  classpath + gsonPath + File.pathSeparator;
        classpath =  classpath + jnaPath + File.pathSeparator;
        classpath =  classpath + jnaPlatformPath + File.pathSeparator;
        classpath =  classpath + snakeYaml + File.pathSeparator;

        return classpath;
    }
	
	/**
	 * Method that gets the path to the JAR from where a specific class is being loaded
	 * @param clazz
	 * 	class of interest
	 * @return the path to the JAR that contains the class
	 * @throws UnsupportedEncodingException if the url of the JAR is not encoded in UTF-8
	 */
	private static String getPathFromClass(Class<?> clazz) throws UnsupportedEncodingException {
	    String classResource = clazz.getName().replace('.', '/') + ".class";
	    URL resourceUrl = clazz.getClassLoader().getResource(classResource);
	    if (resourceUrl == null) {
	        return null;
	    }
	    String urlString = resourceUrl.toString();
	    if (urlString.startsWith("jar:")) {
	        urlString = urlString.substring(4);
	    }
	    if (urlString.startsWith("file:/") && PlatformDetection.isWindows()) {
	        urlString = urlString.substring(6);
	    } else if (urlString.startsWith("file:/") && !PlatformDetection.isWindows()) {
	        urlString = urlString.substring(5);
	    }
	    urlString = URLDecoder.decode(urlString, "UTF-8");
	    File file = new File(urlString);
	    String path = file.getAbsolutePath();
	    if (path.lastIndexOf(".jar!") != -1)
	    	path = path.substring(0, path.lastIndexOf(".jar!")) + ".jar";
	    return path;
	}
}
