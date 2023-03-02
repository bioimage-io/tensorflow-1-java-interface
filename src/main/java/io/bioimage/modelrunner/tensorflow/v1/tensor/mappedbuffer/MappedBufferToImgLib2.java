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

package io.bioimage.modelrunner.tensorflow.v1.tensor.mappedbuffer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

import io.bioimage.modelrunner.tensorflow.v1.tensor.mappedbuffer.ImgLib2ToMappedBuffer;

/**
 * A {@link Img} builder from {@link ByteBuffer} objects
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class MappedBufferToImgLib2
{
	/**
	 * Pattern that matches the header of the temporal file for interprocess communication
	 * and retrieves data type, shape, name and axes
	 */
    private static final Pattern HEADER_PATTERN = Pattern.compile("'dtype':'([a-zA-Z0-9]+)'"
            + ",'axes':'([a-zA-Z]+)'"
            + ",'name':'([^']*)'"
            + ",'shape':'(\\[\\s*(?:(?:[1-9]\\d*|0)\\s*,\\s*)*(?:[1-9]\\d*|0)?\\s*\\])'");
    /**
     * Key for data type info
     */
    private static final String DATA_TYPE_KEY = "dtype";
    /**
     * Key for shape info
     */
    private static final String SHAPE_KEY = "shape";
    /**
     * Key for axes info
     */
    private static final String AXES_KEY = "axes";
    /**
     * Key for axes info
     */
    private static final String NAME_KEY = "name";

    /**
     * Not used (Utility class).
     */
    private MappedBufferToImgLib2()
    {
    }

    /**
     * Creates a {@link Tensor} from the information stored in a {@link ByteBuffer}
     * 
     * @param <T>
     * 	the type of the generated tensor
     * @param buff
     * 	byte buffer to get the tensor info from
     * @return the tensor generated from the bytebuffer
     * @throws IllegalArgumentException if the data type of the tensor saved in the bytebuffer is
     * not supported
     */
    @SuppressWarnings("unchecked")
    public static < T extends RealType< T > & NativeType< T > > Tensor<T> buildTensor(ByteBuffer buff) throws IllegalArgumentException
    {
    	String infoStr = getTensorInfoFromBuffer(buff);
    	HashMap<String, Object> map = getInfoFromHeaderString(infoStr);
    	String dtype = (String) map.get(DATA_TYPE_KEY);
    	String axes = (String) map.get(AXES_KEY);
    	String name = (String) map.get(NAME_KEY);
    	long[] shape = (long[]) map.get(SHAPE_KEY);
    	if (shape.length == 0)
    		return Tensor.buildEmptyTensor(name, axes);
    	
        Img<?> data;
		switch (dtype)
        {
            case "byte":
                data = (Img<?>) buildFromTensorByte(buff, shape);
                break;
            case "int32":
            	data = (Img<?>) buildFromTensorInt(buff, shape);
                break;
            case "float32":
            	data = (Img<?>) buildFromTensorFloat(buff, shape);
                break;
            case "float64":
            	data = (Img<?>) buildFromTensorDouble(buff, shape);
                break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
		return Tensor.build(name, axes, (RandomAccessibleInterval<T>) data);
    }

    /**
     * Creates a {@link Img} from the information stored in a {@link ByteBuffer}
     * 
     * @param <T>
     * 	data type of the image
     * @param byteBuff
     *        The bytebyuffer that contains info to create a tenosr or a {@link Img}
     * @return The imglib2 image {@link Img} built from the bytebuffer info.
     * @throws IllegalArgumentException if the data type of the tensor saved in the bytebuffer is
     * not supported
     */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(ByteBuffer byteBuff) throws IllegalArgumentException
    {
    	String infoStr = getTensorInfoFromBuffer(byteBuff);
    	HashMap<String, Object> map = getInfoFromHeaderString(infoStr);
    	String dtype = (String) map.get(DATA_TYPE_KEY);
    	long[] shape = (long[]) map.get(SHAPE_KEY);
    	if (shape.length == 0)
    		return null;
    	
        // Create an INDArray of the same type of the tensor
        switch (dtype)
        {
            case "byte":
                return (Img<T>) buildFromTensorByte(byteBuff, shape);
            case "int32":
                return (Img<T>) buildFromTensorInt(byteBuff, shape);
            case "float32":
                return (Img<T>) buildFromTensorFloat(byteBuff, shape);
            case "float64":
                return (Img<T>) buildFromTensorDouble(byteBuff, shape);
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
    }

    /**
     * Builds a ByteType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<ByteType> buildFromTensorByte(ByteBuffer tensor, long[] tensorShape)
    {
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
        	tensorCursor.get().set(tensor.get());
		}
	 	return outputImg;
	}

    /**
     * Builds a IntType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<IntType> buildFromTensorInt(ByteBuffer tensor, long[] tensorShape)
    {
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
    	byte[] bytes = new byte[4];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			tensor.get(bytes);
			int val = ((int) (bytes[0] << 24)) + ((int) (bytes[1] << 16)) 
					+ ((int) (bytes[2] << 8)) + ((int) (bytes[3]));
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a FloatType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<FloatType> buildFromTensorFloat(ByteBuffer tensor, long[] tensorShape)
    {
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		byte[] bytes = new byte[4];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			tensor.get(bytes);
			float val = ByteBuffer.wrap(bytes).getFloat();
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    /**
     * Builds a DoubleType {@link Img} from the information stored in a byte buffer.
     * The shape of the image that was previously retrieved from the buffer
     * @param tensor
     * 	byte buffer containing the information of the a tenosr, the position in the buffer
     *  should not be at zero but right after the header.
     * @param tensorShape
     * 	shape of the image to generate, it has been retrieved from the byte buffer 
     * @return image specified in the bytebuffer
     */
    private static Img<DoubleType> buildFromTensorDouble(ByteBuffer tensor, long[] tensorShape)
    {
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
    	byte[] bytes = new byte[8];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			tensor.get(bytes);
			double val = ByteBuffer.wrap(bytes).getDouble();
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }
    
    /**
     * Method that returns the information about the tensor specified at the 
     * beginning of the {@link ByteBuffer} object created 
     * with {@link ImgLib2ToMappedBuffer#build()}.
     * This method reads the buffer from the beginning
     * @param buff
     * 	ByteBuffer containing the information about the tensor
     * @return map containing the name, axes order, datatype and shape of the tensor
     * stored in teh buffer
     */
    public static HashMap<String, Object> readHeaderAndGetInfo(ByteBuffer buff) {
    	buff.clear();
    	return getInfoFromHeaderString(getTensorInfoFromBuffer(buff));
    }
    
    /**
     * GEt the String info stored at the beginning of the buffer that contains
     * the data type, name of tensor, axes and shape info.
     * @param buff
     * 	buffer containing all the data to generate a tensor
     * @return the String header of teh bytebuffer that contains the data about
     *  the tensor (name, data type, shape and axes)
     */
    private static String getTensorInfoFromBuffer(ByteBuffer buff) {
    	byte[] arr = new byte[ImgLib2ToMappedBuffer.MODEL_RUNNER_HEADER.length];
    	buff.get(arr);
    	if (!Arrays.equals(arr, ImgLib2ToMappedBuffer.MODEL_RUNNER_HEADER))
            throw new IllegalArgumentException("Error sending tensors between processes.");
        byte[] lenInfoInBytes = new byte[4];
    	buff.get(lenInfoInBytes);
    	int lenInfo = ByteBuffer.wrap(lenInfoInBytes).getInt();
    	byte[] stringInfoBytes = new byte[lenInfo];
    	buff.get(stringInfoBytes);
        return new String(stringInfoBytes, StandardCharsets.UTF_8);
    }
    
    /**
     * MEthod that retrieves the data type string and shape long array representing
     * the data type and dimensions of the tensor saved in the temp file
     * @param infoStr
     * 	string header of the file that contains the info about the tensor
     * @return dictionary containins the name, dtype, shape and axes of the tensor
     */
    private static HashMap<String, Object> getInfoFromHeaderString(String infoStr) {
       Matcher matcher = HEADER_PATTERN.matcher(infoStr);
       if (!matcher.find()) {
           throw new IllegalArgumentException("Cannot find datatype, name, axes and dimensions "
           		+ "info in file header: " + infoStr);
       }
       String typeStr = matcher.group(1);
       String axesStr = matcher.group(2);
       String nameStr = matcher.group(3);
       String shapeStr = matcher.group(4);
       
       long[] shape = new long[0];
       if (!shapeStr.isEmpty() && !shapeStr.equals("[]")) {
    	   shapeStr = shapeStr.substring(1, shapeStr.length() - 1);
           String[] tokens = shapeStr.split(", ?");
           shape = Arrays.stream(tokens).mapToLong(Long::parseLong).toArray();
       }
       HashMap<String, Object> map = new HashMap<String, Object>();
       map.put(DATA_TYPE_KEY, typeStr);
       map.put(AXES_KEY, axesStr);
       map.put(SHAPE_KEY, shape);
       map.put(NAME_KEY, nameStr);
       return map;
   }
}
