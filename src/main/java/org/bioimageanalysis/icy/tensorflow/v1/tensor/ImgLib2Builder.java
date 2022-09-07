package org.bioimageanalysis.icy.tensorflow.v1.tensor;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link Img} builder for TensorFlow {@link Tensor} objects.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class ImgLib2Builder
{

    /**
     * Not used (Utility class).
     */
    private ImgLib2Builder()
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
    public static <T extends Type<T>> Img<T> build(Tensor<?> tensor) throws IllegalArgumentException
    {
        // Create an INDArray of the same type of the tensor
        switch (tensor.dataType())
        {
            case UINT8:
                return (Img<T>) buildFromTensorByte((Tensor<UInt8>) tensor);
            case INT32:
                return (Img<T>) buildFromTensorInt((Tensor<Integer>) tensor);
            case FLOAT:
                return (Img<T>) buildFromTensorFloat((Tensor<Float>) tensor);
            case DOUBLE:
                return (Img<T>) buildFromTensorDouble((Tensor<Double>) tensor);
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType());
        }
    }

    /**
     * Builds a {@link Img} from a unsigned byte-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static <T extends Type<T>> Img<ByteType> buildFromTensorByte(Tensor<UInt8> tensor)
    {
    	long[] tensorShape = tensor.shape();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        byte[] flatImageArray = new byte[totalSize];
		ByteBuffer outBuff = ByteBuffer.wrap(flatImageArray);
	 	tensor.writeTo(outBuff);
	 	outBuff = null;
		return ArrayImgs.bytes(flatImageArray, tensorShape);
	}

    /**
     * Builds a {@link Img} from a unsigned integer-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The sequence built from the tensor of type {@link DataType#INT}.
     */
    private static <T extends Type<T>> Img<IntType> buildFromTensorInt(Tensor<Integer> tensor)
    {
		long[] tensorShape = tensor.shape();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
    	int[] flatImageArray = new int[totalSize];
    	IntBuffer outBuff = IntBuffer.wrap(flatImageArray);
	 	tensor.writeTo(outBuff);
	 	outBuff = null;
		return ArrayImgs.ints(flatImageArray, tensorShape);
    }

    /**
     * Builds a {@link Img} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static <T extends Type<T>> Img<FloatType> buildFromTensorFloat(Tensor<Float> tensor)
    {
		long[] tensorShape = tensor.shape();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
		float[] flatImageArray = new float[totalSize];
		FloatBuffer outBuff = FloatBuffer.wrap(flatImageArray);
	 	tensor.writeTo(outBuff);
	 	outBuff = null;
		return ArrayImgs.floats(flatImageArray, tensorShape);
    }

    /**
     * Builds a {@link Img} from a unsigned double-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static <T extends Type<T>> Img<DoubleType> buildFromTensorDouble(Tensor<Double> tensor)
    {
		long[] tensorShape = tensor.shape();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
		double[] flatImageArray = new double[totalSize];
		DoubleBuffer outBuff = DoubleBuffer.wrap(flatImageArray);
	 	tensor.writeTo(outBuff);
	 	outBuff = null;
		return ArrayImgs.doubles(flatImageArray, tensorShape);
    }
}
