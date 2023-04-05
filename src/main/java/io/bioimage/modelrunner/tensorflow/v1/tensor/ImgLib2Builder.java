/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java API for Tensorflow 1.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.tensorflow.v1.tensor;

import io.bioimage.modelrunner.utils.IndexingUtils;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

/**
 * A {@link Img} builder for TensorFlow {@link Tensor} objects.
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
	 * Creates a {@link Img} from a given {@link Tensor}
	 * 
	 * @param <T> the type of the image
	 * @param tensor The tensor data is read from.
	 * @return The Img built from the tensor.
	 * @throws IllegalArgumentException If the tensor type is not supported.
	 */
	@SuppressWarnings("unchecked")
	public static <T extends Type<T>> Img<T> build(Tensor<?> tensor)
		throws IllegalArgumentException
	{
		// Create an Img of the same type of the tensor
		switch (tensor.dataType()) {
			case UINT8:
				return (Img<T>) buildFromTensorUByte((Tensor<UInt8>) tensor);
			case INT32:
				return (Img<T>) buildFromTensorInt((Tensor<Integer>) tensor);
			case FLOAT:
				return (Img<T>) buildFromTensorFloat((Tensor<Float>) tensor);
			case DOUBLE:
				return (Img<T>) buildFromTensorDouble((Tensor<Double>) tensor);
			default:
				throw new IllegalArgumentException("Unsupported tensor type: " + tensor
					.dataType());
		}
	}

	/**
	 * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The Img built from the tensor, of type {@link UnsignedByteType}.
	 */
	private static Img<UnsignedByteType> buildFromTensorUByte(Tensor<UInt8> tensor) {
		long[] tensorShape = tensor.shape();
		final ArrayImgFactory<UnsignedByteType> factory = new ArrayImgFactory<>(new UnsignedByteType());
		final Img<UnsignedByteType> outputImg = factory.create(tensorShape);
		Cursor<UnsignedByteType> tensorCursor = outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {
			totalSize *= i;
		}
		byte[] flatArr = new byte[totalSize];
		ByteBuffer outBuff = ByteBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			byte val = flatArr[flatPos];
			if (val < 0)
				tensorCursor.get().set(256 + (int) val);
			else
				tensorCursor.get().set(val);
		}
		return outputImg;
	}

	/**
	 * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The sequence built from the tensor, of type {@link IntType}.
	 */
	private static Img<IntType> buildFromTensorInt(Tensor<Integer> tensor) {
		long[] tensorShape = tensor.shape();
		final ArrayImgFactory<IntType> factory = new ArrayImgFactory<>(new IntType());
		final Img<IntType> outputImg = factory.create(tensorShape);
		Cursor<IntType> tensorCursor = outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {
			totalSize *= i;
		}
		int[] flatArr = new int[totalSize];
		IntBuffer outBuff = IntBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			int val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
	}

	/**
	 * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The Img built from the tensor, of type {@link FloatType}.
	 */
	private static Img<FloatType> buildFromTensorFloat(Tensor<Float> tensor) {
		long[] tensorShape = tensor.shape();
		final ArrayImgFactory<FloatType> factory = new ArrayImgFactory<>(new FloatType());
		final Img<FloatType> outputImg = factory.create(tensorShape);
		Cursor<FloatType> tensorCursor = outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {
			totalSize *= i;
		}
		float[] flatArr = new float[totalSize];
		FloatBuffer outBuff = FloatBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			float val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
	}

	/**
	 * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
	 * 
	 * @param tensor The tensor data is read from.
	 * @return The Img built from the tensor, of type {@link DoubleType}.
	 */
	private static Img<DoubleType> buildFromTensorDouble(Tensor<Double> tensor) {
		long[] tensorShape = tensor.shape();
		final ArrayImgFactory<DoubleType> factory = new ArrayImgFactory<>(new DoubleType());
		final Img<DoubleType> outputImg = factory.create(tensorShape);
		Cursor<DoubleType> tensorCursor = outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {
			totalSize *= i;
		}
		double[] flatArr = new double[totalSize];
		DoubleBuffer outBuff = DoubleBuffer.wrap(flatArr);
		tensor.writeTo(outBuff);
		outBuff = null;
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			double val = flatArr[flatPos];
			tensorCursor.get().set(val);
		}
		return outputImg;
	}
}
