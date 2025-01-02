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

import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.util.Cast;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.UUID;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

/**
 * Utility class to build Tensorflow tensors from shm segments using {@link SharedMemoryArray}
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class TensorBuilder {

	/**
	 * Utility class.
	 */
	private TensorBuilder() {}

	/**
	 * Creates {@link Tensor} instance from a {@link SharedMemoryArray}
	 * 
	 * @param array
	 * 	the {@link SharedMemoryArray} that is going to be converted into
	 *  a {@link Tensor} tensor
	 * @return the Tensorflow {@link Tensor} as the one stored in the shared memory segment
	 * @throws IllegalArgumentException if the type of the {@link SharedMemoryArray}
	 *  is not supported
	 */
	public static Tensor<?> build(SharedMemoryArray array) throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (array.getOriginalDataType().equals("uint8")) {
			return buildUByte(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("int32")) {
			return buildInt(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("float32")) {
			return buildFloat(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("float64")) {
			return buildDouble(Cast.unchecked(array));
		}
		else if (array.getOriginalDataType().equals("int64")) {
			return buildLong(Cast.unchecked(array));
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + array.getOriginalDataType());
		}
	}

	private static Tensor<UInt8> buildUByte(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		Tensor<UInt8> ndarray = Tensor.create(UInt8.class, ogShape, buff);
		return ndarray;
	}

	private static Tensor<Integer> buildInt(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		IntBuffer intBuff = buff.asIntBuffer();
		Tensor<Integer> ndarray = Tensor.create(ogShape, intBuff);
		return ndarray;
	}

	private static Tensor<Long> buildLong(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		LongBuffer longBuff = buff.asLongBuffer();
		Tensor<Long> ndarray = Tensor.create(ogShape, longBuff);
		return ndarray;
	}

	private static Tensor<Float> buildFloat(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		try (FileOutputStream fos = new FileOutputStream("/home/carlos/git/interp_inp" + UUID.randomUUID().toString() + ".npy");
	             FileChannel fileChannel = fos.getChannel()) {
				ByteBuffer buffer = tensor.getDataBuffer();
	            // Write the buffer's content to the file
	            while (buffer.hasRemaining()) {
	                fileChannel.write(buffer);
	            }
	            buffer.rewind();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	        
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		FloatBuffer floatBuff = buff.asFloatBuffer();
		Tensor<Float> ndarray = Tensor.create(ogShape, floatBuff);
		return ndarray;
	}

	private static Tensor<Double> buildDouble(SharedMemoryArray tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.getOriginalShape();
		if (CommonUtils.int32Overflows(ogShape, 1))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per ubyte tensor supported: " + Integer.MAX_VALUE);
		if (!tensor.isNumpyFormat())
			throw new IllegalArgumentException("Shared memory arrays must be saved in numpy format.");
		ByteBuffer buff = tensor.getDataBufferNoHeader();
		DoubleBuffer doubleBuff = buff.asDoubleBuffer();
		Tensor<Double> ndarray = Tensor.create(ogShape, doubleBuff);
		return ndarray;
	}
}
