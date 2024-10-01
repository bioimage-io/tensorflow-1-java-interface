package io.bioimage.modelrunner.tensorflow.v1;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import io.bioimage.modelrunner.apposed.appose.Types;
import io.bioimage.modelrunner.apposed.appose.Service.RequestType;
import io.bioimage.modelrunner.apposed.appose.Service.ResponseType;

public class JavaWorker {
	
	private static LinkedHashMap<String, Object> tasks = new LinkedHashMap<String, Object>();
	
	private final String uuid;
	
	private final Tensorflow1Interface ti;
		
	private boolean cancelRequested = false;

	/**
	 * Method in the child process that is in charge of keeping the process open and calling the model load,
	 * model inference and model closing
	 * @param args
	 * 	args of the parent process
	 */
	public static void main(String[] args) {
    	
    	try(Scanner scanner = new Scanner(System.in)){
    		Tensorflow1Interface ti;
    		try {
				ti = new Tensorflow1Interface(false);
			} catch (IOException | URISyntaxException e) {
				return;
			}
            
            while (true) {
            	String line;
                try {
                    if (!scanner.hasNextLine()) break;
                    line = scanner.nextLine().trim();
                } catch (Exception e) {
                    break;
                }
                
                if (line.isEmpty()) break;
                Map<String, Object> request = Types.decode(line);
                String uuid = (String) request.get("task");
                String requestType = (String) request.get("requestType");
                
                if (requestType.equals(RequestType.EXECUTE.toString())) {
                	String script = (String) request.get("script");
                	Map<String, Object> inputs = (Map<String, Object>) request.get("inputs");
                	JavaWorker task = new JavaWorker(uuid, ti);
                	tasks.put(uuid, task);
                	task.start(script, inputs);
                } else if (requestType.equals(RequestType.CANCEL.toString())) {
                	JavaWorker task = (JavaWorker) tasks.get(uuid);
                	if (task == null) {
                		System.err.println("No such task: " + uuid);
                		continue;
                	}
                	task.cancelRequested = true;
                } else {
                	break;
                }
            }
    	}
		
	}
	
	private JavaWorker(String uuid, Tensorflow1Interface ti) {
		this.uuid = uuid;
		this.ti = ti;
	}
	
	private void executeScript(String script, Map<String, Object> inputs) {
		Map<String, Object> binding = new LinkedHashMap<String, Object>();
		binding.put("task", this);
		if (inputs != null)
			binding.putAll(binding);
		
		this.reportLaunch();
		try {
			if (script.equals("loadModel")) {
				ti.loadModel((String) inputs.get("modelFolder"), null);
			} else if (script.equals("inference")) {
				ti.runFromShmas((List<String>) inputs.get("inputs"), (List<String>) inputs.get("outputs"));
			} else if (script.equals("close")) {
				ti.closeModel();
			}
		} catch(Exception ex) {
			this.fail(Types.stackTrace(ex));
			return;
		}
		this.reportCompletion();
	}
	
	private void start(String script, Map<String, Object> inputs) {
		new Thread(() -> executeScript(script, inputs), "Appose-" + this.uuid).start();
	}
	
	private void reportLaunch() {
		respond(ResponseType.LAUNCH, null);
	}
	
	private void reportCompletion() {
		respond(ResponseType.COMPLETION, null);
	}
	
	private void update(String message, Integer current, Integer maximum) {
		LinkedHashMap<String, Object> args = new LinkedHashMap<String, Object>();
		
		if (message != null)
			args.put("message", message);
		
		if (current != null)
			args.put("current", current);
		
		if (maximum != null)
			args.put("maximum", maximum);
		this.respond(ResponseType.UPDATE, args);
	}
	
	private void respond(ResponseType responseType, Map<String, Object> args) {
		Map<String, Object> response = new HashMap<String, Object>();
		response.put("task", uuid);
		response.put("responseType", responseType);
		if (args != null)
			response.putAll(args);
		try {
			System.out.println(Types.encode(response));
			System.out.flush();
		} catch(Exception ex) {
			this.fail(Types.stackTrace(ex.getCause()));
		}
	}
	
	private void cancel() {
		this.respond(ResponseType.CANCELATION, null);
	}
	
	private void fail(String error) {
		Map<String, Object> args = null;
		if (error != null) {
			args = new HashMap<String, Object>();
			args.put("error", error);
		}
        respond(ResponseType.FAILURE, args);
	}

}
