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

import com.google.protobuf.InvalidProtocolBufferException;

import io.bioimage.modelrunner.bioimageio.description.ModelDescriptor;
import io.bioimage.modelrunner.bioimageio.download.DownloadModel;
import io.bioimage.modelrunner.engine.DeepLearningEngineInterface;
import io.bioimage.modelrunner.engine.EngineInfo;
import io.bioimage.modelrunner.exceptions.LoadModelException;
import io.bioimage.modelrunner.exceptions.RunModelException;
import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.Tensor;
import io.bioimage.modelrunner.tensorflow.v1.tensor.ImgLib2Builder;
import io.bioimage.modelrunner.tensorflow.v1.tensor.TensorBuilder;
import io.bioimage.modelrunner.tensorflow.v1.tensor.mappedbuffer.ImgLib2ToMappedBuffer;
import io.bioimage.modelrunner.tensorflow.v1.tensor.mappedbuffer.MappedBufferToImgLib2;
import io.bioimage.modelrunner.utils.Constants;
import io.bioimage.modelrunner.utils.ZipUtils;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.security.CodeSource;
import java.security.ProtectionDomain;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
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
	/**
	 * Internal object of the Tensorflow model
	 */
	private static SignatureDef sig;
	/**
	 * Whether the execution needs interprocessing (MacOS Interl) or not
	 */
	private boolean interprocessing = false;
    /**
     * TEmporary dir where to store temporary files
     */
    private String tmpDir;
    /**
     * Folde containing the model that is being executed
     */
    private String modelFolder;
    /**
     * List of temporary files used for interprocessing communication
     */
    private List<File> listTempFiles;
    /**
     * HashMap that maps tensor to the temporal file name for interprocessing
     */
    private HashMap<String, String> tensorFilenameMap;
    
    /**
     * Constructor that detects whether the operating system where it is being 
     * executed is MacOS Intel or not to know if it is going to need interprocessing 
     * or not
     * @throws IOException if the temporary dir is not found
     */
    public Tensorflow1Interface() throws IOException
    {
    	boolean isMac = PlatformDetection.isMacOS();
    	boolean isIntel = PlatformDetection.getArch().equals(PlatformDetection.ARCH_X86_64);
    	if (false && isMac && isIntel) {
    		interprocessing = true;
    		tmpDir = getTemporaryDir();
    		
    	}
    }
	
    /**
     * Private constructor that can only be launched from the class to create a separate
     * process to avoid the conflicts that occur in the same process between TF1 and TF2
     * in MacOS Intel
     * @param doInterprocessing
     * 	whether to do interprocessing or not
     * @throws IOException if the temp dir is not found
     */
    private Tensorflow1Interface(boolean doInterprocessing) throws IOException
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
			return;
		}
		try {
			checkModelUnzipped();
		} catch (Exception e) {
			throw new LoadModelException(e.toString());
		}
		model = SavedModelBundle.load(modelFolder, "serve");
		byte[] byteGraph = model.metaGraphDef();
		try {
			sig = MetaGraphDef.parseFrom(byteGraph).getSignatureDefOrThrow(
				"serving_default");
		}
		catch (InvalidProtocolBufferException e) {
			closeModel();
			throw new LoadModelException(e.toString());
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
		unzipTfWeights(ModelDescriptor.readFromLocalFile(modelFolder + File.separator + Constants.RDF_FNAME));
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
			source = DownloadModel.getFileNameFromURLString(source);
			System.out.println("Unzipping model...");
			ZipUtils.unzipFolder(modelFolder + File.separator + source, modelFolder);
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
		fillOutputTensors(resultPatchTensors, outputTensors);
		for (org.tensorflow.Tensor<?> tt : inTensors) {
			tt.close();
		}
		for (org.tensorflow.Tensor<?> tt : resultPatchTensors) {
			tt.close();
		}
	}
	
	/**
	 * MEthod only used in MacOS Intel systems that makes all the arangements
	 * to create another process, communicate the model info and tensors to the other 
	 * process and then retrieve the results of the other process
	 * @param inputTensors
	 * 	tensors that are going to be run on the model
	 * @param outputTensors
	 * 	expected results of the model
	 * @throws RunModelException if there is any issue running the model
	 */
	public void runInterprocessing(List<Tensor<?>> inputTensors, List<Tensor<?>> outputTensors) throws RunModelException {
		createTensorsForInterprocessing(inputTensors);
		createTensorsForInterprocessing(outputTensors);
		try {
			List<String> args = getProcessCommandsWithoutArgs();
			for (Tensor tensor : inputTensors) {args.add(getFilename4Tensor(tensor.getName()) + INPUT_FILE_TERMINATION);}
			for (Tensor tensor : outputTensors) {args.add(getFilename4Tensor(tensor.getName()) + OUTPUT_FILE_TERMINATION);}
			ProcessBuilder builder = new ProcessBuilder(args);
			builder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
			builder.redirectError(ProcessBuilder.Redirect.INHERIT);
	        Process process = builder.start();
	        int result = process.waitFor();
	        process.destroy();
	        if (result != 0)
	    		throw new RunModelException("Error executing the Tensorflow 1 model in"
	        			+ " a separate process. The process was not terminated correctly."
	        			+ System.lineSeparator() + readProcessStringOutput(process));
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
	 * @throws RunModelException If the number of tensors expected is not the same
	 *           as the number of Tensors outputed by the model
	 */
	public static void fillOutputTensors(
		List<org.tensorflow.Tensor<?>> outputNDArrays,
		List<Tensor<?>> outputTensors) throws RunModelException
	{
		if (outputNDArrays.size() != outputTensors.size())
			throw new RunModelException(outputNDArrays.size(), outputTensors.size());
		for (int i = 0; i < outputNDArrays.size(); i++) {
			try {
				outputTensors.get(i).setData(ImgLib2Builder.build(outputNDArrays.get(i)));
			} catch (IllegalArgumentException ex) {
				throw new RunModelException(ex.toString());
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
		sig = null;
		if (model != null) {
			model.session().close();
			model.close();
		}
		model = null;
		if (listTempFiles == null)
			return;
		for (File ff : listTempFiles) {
			if (ff.exists())
				ff.delete();
		}
		listTempFiles = null;
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
	 * This method checks that the arguments are correct, retrieves the input and output
	 * tensors, loads the model, makes inference with it and finally sends the tensors
	 * to the original process
     * 
     * @param args
     * 	arguments of the program:
     * 		- Path to the model folder
     * 		- Path to a temporary dir
     * 		- Name of the input 0
     * 		- Name of the input 1
     * 		- ...
     * 		- Name of the output n
     * 		- Name of the output 0
     * 		- Name of the output 1
     * 		- ...
     * 		- Name of the output n
     * @throws LoadModelException if there is any error loading the model
     * @throws IOException	if there is any error reading or writing any file or with the paths
     * @throws RunModelException	if there is any error running the model
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
    	
    	HashMap<String, List<String>> map = tfInterface.getInputTensorsFileNames(args);
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
    
    /**
     * Get the name of teh temporary file associated to the tensor name
     * @param name
     * 	name of the tensor
     * @return file name associated to the tensor
     */
    private String getFilename4Tensor(String name) {
    	if (tensorFilenameMap == null)
    		tensorFilenameMap = new HashMap<String, String>();
    	if (tensorFilenameMap.get(name) != null)
    		return tensorFilenameMap.get(name);
    	LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddHHmmssSSS");
    	String newName = name + "_" +  now.format(formatter);
    	tensorFilenameMap.put(name, newName);
		return tensorFilenameMap.get(name);
    }
	
    /**
     * Create a temporary file for each of the tensors in the list to communicate with 
     * the separate process in MacOS Intel systems
     * @param tensors
     * 	list of tensors to be sent
     * @throws RunModelException if there is any error converting the tensors
     */
	private void createTensorsForInterprocessing(List<Tensor<?>> tensors) throws RunModelException{
		if (this.listTempFiles == null)
			this.listTempFiles = new ArrayList<File>();
		for (Tensor<?> tensor : tensors) {
			long lenFile = ImgLib2ToMappedBuffer.findTotalLengthFile(tensor);
			File ff = new File(tmpDir + File.separator + getFilename4Tensor(tensor.getName()) + FILE_EXTENSION);
			if (!ff.exists()) {
				ff.deleteOnExit();
				this.listTempFiles.add(ff);
			}
			try (RandomAccessFile rd = 
    				new RandomAccessFile(ff, "rw");
    				FileChannel fc = rd.getChannel();) {
    			MappedByteBuffer mem = fc.map(FileChannel.MapMode.READ_WRITE, 0, lenFile);
    			ByteBuffer byteBuffer = mem.duplicate();
    			ImgLib2ToMappedBuffer.build(tensor, byteBuffer);
    		} catch (IOException e) {
    			closeModel();
    			throw new RunModelException(e.getCause().toString());
			}
		}
	}
	
	/**
	 * Retrieves the data of the tensors contained in the input list from the output
	 * generated by the independent process
	 * @param tensors
	 * 	list of tensors that are going to be filled
	 * @throws RunModelException if there is any issue retrieving the data from the other process
	 */
	private void retrieveInterprocessingTensors(List<Tensor<?>> tensors) throws RunModelException{
		for (Tensor<?> tensor : tensors) {
			try (RandomAccessFile rd = 
    				new RandomAccessFile(tmpDir + File.separator 
    						+ this.getFilename4Tensor(tensor.getName()) + FILE_EXTENSION, "r");
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
	
	/**
	 * Create a tensor from the data contained in a file named as the parameter
	 * provided as an input + the file extension {@link #FILE_EXTENSION}.
	 * This file is produced by another process to communicate with the current process
	 * @param <T>
	 * 	generic type of the tensor
	 * @param name
	 * 	name of the file without the extension ({@link #FILE_EXTENSION}).
	 * @return a tensor created with the data in the file
	 * @throws RunModelException if there is any problem retrieving the data and cerating the tensor
	 */
	private < T extends RealType< T > & NativeType< T > > Tensor<T> 
				retrieveInterprocessingTensorsByName(String name) throws RunModelException {
		try (RandomAccessFile rd = 
				new RandomAccessFile(tmpDir + File.separator 
						+ this.getFilename4Tensor(name) + FILE_EXTENSION, "r");
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

        String modelrunnerPath = getPathFromClass(DeepLearningEngineInterface.class);
        String imglib2Path = getPathFromClass(NativeType.class);
        if (modelrunnerPath == null || (modelrunnerPath.endsWith("DeepLearningEngineInterface.class") 
        		&& !modelrunnerPath.contains(File.pathSeparator)))
        	modelrunnerPath = System.getProperty("java.class.path");
        String classpath =  modelrunnerPath + File.pathSeparator + imglib2Path + File.pathSeparator;
        ProtectionDomain protectionDomain = Tensorflow1Interface.class.getProtectionDomain();
        String codeSource = protectionDomain.getCodeSource().getLocation().getPath();
        String f_name = URLDecoder.decode(codeSource, StandardCharsets.UTF_8.toString());
	        for (File ff : new File(f_name).getParentFile().listFiles()) {
	        	classpath += ff.getAbsolutePath() + File.pathSeparator;
	        }
        String className = Tensorflow1Interface.class.getName();
        List<String> command = new LinkedList<String>();
        command.add(padSpecialJavaBin(javaBin));
        command.add("-cp");
        command.add(classpath);
        command.add(className);
        command.add(modelFolder);
        command.add(this.tmpDir);
        return command;
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
	
	/**
	 * Get temporary directory to perform the interprocessing communication in MacOSX intel
	 * @return the tmp dir
	 * @throws IOException if the files cannot be written in any of the temp dirs
	 */
	private static String getTemporaryDir() throws IOException {
		String tmpDir;
		String enginesDir = getEnginesDir();
		if (enginesDir != null && Files.isWritable(Paths.get(enginesDir))) {
			tmpDir = enginesDir + File.separator + "temp";
			if (!(new File(tmpDir).isDirectory()) &&  !(new File(tmpDir).mkdirs()))
				tmpDir = enginesDir;
		} else if (System.getenv("temp") != null
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
		} else {
			throw new IOException("Unable to find temporal directory with writting rights. "
					+ "Please either allow writting on the system temporal folder or on '" + enginesDir + "'.");
		}
		return tmpDir;
	}
	
	/**
	 * GEt the directory where the TF2 engine is located if a temporary dir is not found
	 * @return directory of the engines
	 */
	private static String getEnginesDir() {
		String dir;
		try {
			dir = getPathFromClass(Tensorflow1Interface.class);
		} catch (UnsupportedEncodingException e) {
			String classResource = Tensorflow1Interface.class.getName().replace('.', '/') + ".class";
		    URL resourceUrl = Tensorflow1Interface.class.getClassLoader().getResource(classResource);
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
		    File file = new File(urlString);
		    String path = file.getAbsolutePath();
		    if (path.lastIndexOf(".jar!") != -1)
		    	path = path.substring(0, path.lastIndexOf(".jar!")) + ".jar";
		    dir = path;
		}
		return new File(dir).getParent();
	}
    
    /**
     * Retrieve the file names used for interprocess communication
     * @param args
     * 	args provided to the main method
     * @return a map with a list of input and output names
     */
    private HashMap<String, List<String>> getInputTensorsFileNames(String[] args) {
    	List<String> inputNames = new ArrayList<String>();
    	List<String> outputNames = new ArrayList<String>();
    	if (this.tensorFilenameMap == null)
    		this.tensorFilenameMap = new HashMap<String, String>();
    	for (int i = 2; i < args.length; i ++) {
    		if (args[i].endsWith(INPUT_FILE_TERMINATION)) {
    			String nameWTimestamp = args[i].substring(0, args[i].length() - INPUT_FILE_TERMINATION.length());
    			String onlyName = nameWTimestamp.substring(0, nameWTimestamp.lastIndexOf("_"));
    			inputNames.add(onlyName);
    			tensorFilenameMap.put(onlyName, nameWTimestamp);
    		} else if (args[i].endsWith(OUTPUT_FILE_TERMINATION)) {
    			String nameWTimestamp = args[i].substring(0, args[i].length() - OUTPUT_FILE_TERMINATION.length());
    			String onlyName = nameWTimestamp.substring(0, nameWTimestamp.lastIndexOf("_"));
    			outputNames.add(onlyName);
    			tensorFilenameMap.put(onlyName, nameWTimestamp);
    	
    		}
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
    
    /**
     * MEthod to obtain the String output of the process in case something goes wrong
     * @param process
     * 	the process that executed the TF1 model
     * @return the String output that we would have seen on the terminal
     * @throws IOException if the output of the terminal cannot be seen
     */
    private static String readProcessStringOutput(Process process) throws IOException {
    	BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
		BufferedReader bufferedErrReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
		String text = "";
		String line;
	    while ((line = bufferedErrReader.readLine()) != null) {
	    	text += line + System.lineSeparator();
	    }
	    while ((line = bufferedReader.readLine()) != null) {
	    	text += line + System.lineSeparator();
	    }
	    return text;
    }
}
