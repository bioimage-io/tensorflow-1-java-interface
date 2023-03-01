package io.bioimage.modelrunner.tensorflow.v1.tensor;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import io.bioimage.modelrunner.tensor.Tensor;
import net.imglib2.Cursor;
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

/**
 * A {@link Img} builder for TensorFlow {@link Tensor} objects.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class MappedBufferToImgLib2
{
	/**
	 * Pattern that matches the header of the temporal file for interprocess communication
	 * and retrieves data type and shape
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
     * Creates a {@link Img} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    @SuppressWarnings("unchecked")
    public static < T extends RealType< T > & NativeType< T > > Tensor<T> buildTensor(ByteBuffer buff) throws IllegalArgumentException
    {
    	String infoStr = getTensorInfoFromBuffer(buff);
    	HashMap<String, Object> map = getDataTypeAndShape(infoStr);
    	String dtype = (String) map.get(DATA_TYPE_KEY);
    	String axes = (String) map.get(AXES_KEY);
    	String name = (String) map.get(NAME_KEY);
    	long[] shape = (long[]) map.get(SHAPE_KEY);
    	if (shape.length == 0)
    		return Tensor.buildEmptyTensor(name, axes);
    	
        Img<T> data;
		switch (dtype)
        {
            case "byte":
                data = (Img<T>) buildFromTensorByte(buff, shape);
                break;
            case "int32":
            	data = (Img<T>) buildFromTensorInt(buff, shape);
                break;
            case "float32":
            	data = (Img<T>) buildFromTensorFloat(buff, shape);
                break;
            case "float64":
            	data = (Img<T>) buildFromTensorDouble(buff, shape);
                break;
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
		return Tensor.build(name, axes, data);
    }

    /**
     * Creates a {@link Img} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    @SuppressWarnings("unchecked")
    public static <T extends Type<T>> Img<T> build(ByteBuffer tensor) throws IllegalArgumentException
    {
    	String infoStr = getTensorInfoFromBuffer(tensor);
    	HashMap<String, Object> map = getDataTypeAndShape(infoStr);
    	String dtype = (String) map.get(DATA_TYPE_KEY);
    	long[] shape = (long[]) map.get(SHAPE_KEY);
    	if (shape.length == 0)
    		return null;
    	
        // Create an INDArray of the same type of the tensor
        switch (dtype)
        {
            case "byte":
                return (Img<T>) buildFromTensorByte(tensor, shape);
            case "int32":
                return (Img<T>) buildFromTensorInt(tensor, shape);
            case "float32":
                return (Img<T>) buildFromTensorFloat(tensor, shape);
            case "float64":
                return (Img<T>) buildFromTensorDouble(tensor, shape);
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + dtype);
        }
    }

    /**
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
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
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
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
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
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
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
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
     * GEt the String info stored at the beginning of the buffer that contains
     * the data type and the shape.
     * @param buff
     * @return
     */
    public static String getTensorInfoFromBuffer(ByteBuffer buff) {
    	byte[] arr = new byte[MappedFileBuilder.MODEL_RUNNER_HEADER.length];
    	buff.get(arr);
    	if (!Arrays.equals(arr, MappedFileBuilder.MODEL_RUNNER_HEADER))
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
     * @return
     */
    public static HashMap<String, Object> getDataTypeAndShape(String infoStr) {
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
