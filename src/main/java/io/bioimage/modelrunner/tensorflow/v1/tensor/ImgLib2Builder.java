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
package io.bioimage.modelrunner.tensorflow.v1.tensor;

import io.bioimage.modelrunner.tensor.Utils;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

/**
 * A {@link RandomAccessibleInterval} builder for TensorFlow {@link Tensor} objects.
 * Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
 * from Tensorflow 1 {@link Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ImgLib2Builder {

	/**
	 * Not used (Utility class).
	 */
	private ImgLib2Builder() {}

	/**
	 * Creates a {@link RandomAccessibleInterval} from a given {@link Tensor}
	 * 
	 * @param <T> the type of the image
	 * @param tensor The tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor.
	 * @throws IllegalArgumentException If the tensor type is not supported.
	 */
	@SuppressWarnings("unchecked")
	public static < T extends Type< T > >  RandomAccessibleInterval<T> build(Tensor<?> tensor)
		throws IllegalArgumentException
	{
		// Create an Img of the same type of the tensor
		switch (tensor.dataType()) {
			case UINT8:
				return (RandomAccessibleInterval<T>) buildFromTensorUByte((Tensor<UInt8>) tensor);
			case INT32:
				return (RandomAccessibleInterval<T>) buildFromTensorInt((Tensor<Integer>) tensor);
			case FLOAT:
				return (RandomAccessibleInterval<T>) buildFromTensorFloat((Tensor<Float>) tensor);
			case DOUBLE:
				return (RandomAccessibleInterval<T>) buildFromTensorDouble((Tensor<Double>) tensor);
			default:
				throw new IllegalArgumentException("Unsupported tensor type: " + tensor
					.dataType());
		}
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned byte-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor, of type {@link UnsignedByteType}.
	 */
	private static RandomAccessibleInterval<UnsignedByteType> buildFromTensorUByte(Tensor<UInt8> tensor) {
		long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		byte[] flatArr = new byte[totalSize];
		ByteBuffer outBuff = ByteBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;
		RandomAccessibleInterval<UnsignedByteType> rai = ArrayImgs.unsignedBytes(flatArr, tensorShape);
		return Utils.transpose(rai);
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned integer-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor, of type {@link IntType}.
	 */
	private static RandomAccessibleInterval<IntType> buildFromTensorInt(Tensor<Integer> tensor) {
		long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		int[] flatArr = new int[totalSize];
		IntBuffer outBuff = IntBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;
		RandomAccessibleInterval<IntType> rai = ArrayImgs.ints(flatArr, tensorShape);
		return Utils.transpose(rai);
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned float-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor, of type {@link FloatType}.
	 */
	private static RandomAccessibleInterval<FloatType> buildFromTensorFloat(Tensor<Float> tensor) {
		long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		float[] flatArr = new float[totalSize];
		FloatBuffer outBuff = FloatBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;;
		RandomAccessibleInterval<FloatType> rai = ArrayImgs.floats(flatArr, tensorShape);
		return Utils.transpose(rai);
	}

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned double-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The RandomAccessibleInterval built from the tensor, of type {@link DoubleType}.
	 */
	private static RandomAccessibleInterval<DoubleType> buildFromTensorDouble(Tensor<Double> tensor) {
		long[] arrayShape = tensor.shape();
		long[] tensorShape = new long[arrayShape.length];
		for (int i = 0; i < arrayShape.length; i ++) tensorShape[i] = arrayShape[arrayShape.length - 1 - i];
		int totalSize = 1;
		for (long i : tensorShape) totalSize *= i;
		double[] flatArr = new double[totalSize];
		DoubleBuffer outBuff = DoubleBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;
		RandomAccessibleInterval<DoubleType> rai = ArrayImgs.doubles(flatArr, tensorShape);
		return Utils.transpose(rai);
	}
}
