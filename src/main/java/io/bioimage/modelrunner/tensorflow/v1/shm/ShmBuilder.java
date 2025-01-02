/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
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
package io.bioimage.modelrunner.tensorflow.v1.shm;

import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.UUID;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A utility class that converts {@link Tensor} tensors into {@link SharedMemoryArray}s for
 * interprocessing communication
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ShmBuilder
{
    /**
     * Utility class.
     */
    private ShmBuilder()
    {
    }

    /**
     * Create a {@link SharedMemoryArray} from a {@link Tensor} tensor
     * @param tensor
     * 	the tensor to be passed into the other process through the shared memory
     * @param memoryName
     * 	the name of the memory region where the tensor is going to be copied
     * @throws IllegalArgumentException if the data type of the tensor is not supported
     * @throws IOException if there is any error creating the shared memory array
     */
    @SuppressWarnings("unchecked")
	public static void build(Tensor<?> tensor, String memoryName) throws IllegalArgumentException, IOException
    {
		switch (tensor.dataType())
        {
            case UINT8:
            	buildFromTensorUByte((Tensor<UInt8>) tensor, memoryName);
            	break;
            case INT32:
            	buildFromTensorInt((Tensor<Integer>) tensor, memoryName);
            	break;
            case FLOAT:
            	buildFromTensorFloat((Tensor<Float>) tensor, memoryName);
            	break;
            case DOUBLE:
            	buildFromTensorDouble((Tensor<Double>) tensor, memoryName);
            	break;
            case INT64:
            	buildFromTensorLong((Tensor<Long>) tensor, memoryName);
            	break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

    private static void buildFromTensorUByte(Tensor<UInt8> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 1))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per ubyte output tensor supported: " + Integer.MAX_VALUE / 1);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new UnsignedByteType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.writeTo(buff);
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorInt(Tensor<Integer> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per int output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.writeTo(buff);
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorFloat(Tensor<Float> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per float output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new FloatType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.writeTo(buff);
        try (FileOutputStream fos = new FileOutputStream("/home/carlos/git/interp_out" + UUID.randomUUID().toString() + ".npy");
	             FileChannel fileChannel = fos.getChannel()) {
            	buff.rewind();
	            // Write the buffer's content to the file
	            while (buff.hasRemaining()) {
	                fileChannel.write(buff);
	            }
	            buff.rewind();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorDouble(Tensor<Double> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per double output tensor supported: " + Integer.MAX_VALUE / 8);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new DoubleType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.writeTo(buff);
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorLong(Tensor<Long> tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per long output tensor supported: " + Integer.MAX_VALUE / 8);
		

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new LongType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.writeTo(buff);
        if (PlatformDetection.isWindows()) shma.close();
    }
}
