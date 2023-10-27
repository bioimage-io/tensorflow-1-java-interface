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
import net.imglib2.blocks.PrimitiveBlocks;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

/**
 * A TensorFlow {@link Tensor} builder for {@link Img} and
 * {@link io.bioimage.modelrunner.tensor.Tensor} objects.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class TensorBuilder {

	/**
	 * Not used (Utility class).
	 */
	private TensorBuilder() {}

	/**
	 * Creates a {@link Tensor} based on the provided
	 * {@link io.bioimage.modelrunner.tensor.Tensor} and the desired dimension
	 * order for the resulting tensor.
	 * 
	 * @param tensor The Tensor to be converted.
	 * @return The tensor created from the sequence.
	 * @throws IllegalArgumentException If the ndarray type is not supported.
	 */
	public static < T extends RealType< T > & NativeType< T > > 
		Tensor<?> build(io.bioimage.modelrunner.tensor.Tensor<T> tensor) {
		return build(tensor.getData());
	}

	/**
	 * Creates a {@link Tensor} based on the provided
	 * {@link RandomAccessibleInterval} and the desired dimension order for the
	 * resulting tensor.
	 * 
	 * @param rai The {@link RandomAccessibleInterval} to be converted.
	 * @return The tensor created from the sequence.
	 * @throws IllegalArgumentException If the {@link RandomAccessibleInterval} dtype is not supported.
	 */
	public static  Tensor<?> build(RandomAccessibleInterval<?> rai)
	{
		if (Util.getTypeFromInterval(rai) instanceof ByteType) {
			return buildByte((RandomAccessibleInterval<ByteType>) rai);
		}
		else if (Util.getTypeFromInterval(rai) instanceof IntType) {
			return buildInt((RandomAccessibleInterval<IntType>) rai);
		}
		else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
			return buildFloat((RandomAccessibleInterval<FloatType>) rai);
		}
		else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
			return buildDouble((RandomAccessibleInterval<DoubleType>) rai);
		}
		else {
			throw new IllegalArgumentException("The image has an unsupported type: " +
				Util.getTypeFromInterval(rai).getClass().toString());
		}
	}

	/**
	 * Creates a unsigned byte-typed {@link Tensor} based on the provided
	 * {@link RandomAccessibleInterval} and the desired dimension order for the
	 * resulting tensor.
	 * 
	 * @param tensor The {@link RandomAccessibleInterval} to be converted.
	 * @return The {@link Tensor} created from the sequence.
	 */
	private static Tensor<UInt8> buildByte(
		RandomAccessibleInterval<ByteType> tensor)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< ByteType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
		ByteBuffer buff = ByteBuffer.wrap(flatArr);
		Tensor<UInt8> ndarray = Tensor.create(UInt8.class, ogShape, buff);
		return ndarray;
	}

	/**
	 * Creates a integer-typed {@link Tensor} based on the provided
	 * {@link RandomAccessibleInterval} and the desired dimension order for the
	 * resulting tensor.
	 *  
	 * @param tensor The {@link RandomAccessibleInterval} to be converted.
	 * @return The {@link Tensor} created from the sequence.
	 */
	private static Tensor<Integer> buildInt(
		RandomAccessibleInterval<IntType> tensor)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< IntType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
		IntBuffer buff = IntBuffer.wrap(flatArr);
		Tensor<Integer> ndarray = Tensor.create(ogShape, buff);
		return ndarray;
	}

	/**
	 * Creates a float-typed {@link Tensor} based on the provided
	 * {@link RandomAccessibleInterval} and the desired dimension order for the
	 * resulting tensor.
	 * 
	 * @param tensor The {@link RandomAccessibleInterval} to be converted.
	 * @return The {@link Tensor} created from the sequence.
	 */
	private static Tensor<Float> buildFloat(
		RandomAccessibleInterval<FloatType> tensor)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< FloatType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
		FloatBuffer buff = FloatBuffer.wrap(flatArr);
		Tensor<Float> ndarray = Tensor.create(ogShape, buff);
		return ndarray;
	}

	/**
	 * Creates a double-typed {@link Tensor} based on the provided
	 * {@link RandomAccessibleInterval} and the desired dimension order for the
	 * resulting tensor.
	 * 
	 * @param tensor The {@link RandomAccessibleInterval} to be converted.
	 * @return The {@link Tensor} created from the sequence.
	 */
	private static Tensor<Double> buildDouble(
		RandomAccessibleInterval<DoubleType> tensor)
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		tensor = Utils.transpose(tensor);
		PrimitiveBlocks< DoubleType > blocks = PrimitiveBlocks.of( tensor );
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];
		blocks.copy( new long[tensorShape.length], flatArr, sArr );
		DoubleBuffer buff = DoubleBuffer.wrap(flatArr);
		Tensor<Double> ndarray = Tensor.create(ogShape, buff);
		return ndarray;
	}
}
