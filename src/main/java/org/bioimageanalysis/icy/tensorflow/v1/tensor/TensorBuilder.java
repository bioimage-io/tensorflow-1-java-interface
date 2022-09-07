package org.bioimageanalysis.icy.tensorflow.v1.tensor;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.bioimageanalysis.icy.deeplearning.tensor.RaiArrayUtils;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;

/**
 * A TensorFlow {@link Tensor} builder for {@link INDArray} and {@link org.bioimageanalysis.icy.deeplearning.tensor.Tensor} objects.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public final class TensorBuilder
{

    /**
     * Not used (Utility class).
     */
    private TensorBuilder()
    {
    }

    /**
     * Creates a {@link Tensor} based on the provided {@link org.bioimageanalysis.icy.deeplearning.tensor.Tensor} and the desired dimension order for the resulting tensor.
     * 
     * @param ndarray
     *        The Tensor to be converted.
     * @return The tensor created from the sequence.
     * @throws IllegalArgumentException
     *         If the ndarray type is not supported.
     */
    public static Tensor<?> build(org.bioimageanalysis.icy.deeplearning.tensor.Tensor tensor)
    {
    	return build(tensor.getData());
    }

    /**
     * Creates a {@link Tensor} based on the provided {@link RandomAccessibleInterval} and the desired dimension order for the resulting tensor.
     * 
     * @param rai
     *        The NDArray to be converted.
     * @return The tensor created from the sequence.
     * @throws IllegalArgumentException
     *         If the ndarray type is not supported.
     */
    public static <T extends Type<T>> Tensor<?> build(RandomAccessibleInterval<T> rai)
    {
    	if (Util.getTypeFromInterval(rai) instanceof ByteType) {
    		return buildByte((RandomAccessibleInterval<ByteType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof IntType) {
    		return buildInt((RandomAccessibleInterval<IntType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof FloatType) {
    		return buildFloat((RandomAccessibleInterval<FloatType>) rai);
    	} else if (Util.getTypeFromInterval(rai) instanceof DoubleType) {
    		return buildDouble((RandomAccessibleInterval<DoubleType>) rai);
    	} else {
            throw new IllegalArgumentException("The image has an unsupported type: " + Util.getTypeFromInterval(rai).getClass().toString());
    	}
    }

    /**
     * Creates a unsigned byte-typed {@link Tensor} based on the provided {@link RandomAccessibleInterval} and the desired dimension order for the resulting tensor.
     * 
     * @param ndarray
     *        The sequence to be converted.
     * @return The INDArray created from the sequence.
     * @throws IllegalArgumentException
     *         If the ndarray type is not supported.
     */
    private static <T extends Type<T>>  Tensor<UInt8> buildByte(RandomAccessibleInterval<ByteType> ndarray)
    {
    	byte[] arr = RaiArrayUtils.byteArray(ndarray);
    	ByteBuffer buff = ByteBuffer.wrap(arr);
		Tensor<UInt8> tensor = Tensor.create(UInt8.class, ndarray.dimensionsAsLongArray(), buff);
		return tensor;
    }

    /**
     * Creates a integer-typed {@link Tensor} based on the provided {@link RandomAccessibleInterval} and the desired dimension order for the resulting tensor.
     * 
     * @param ndarray
     *        The sequence to be converted.
     * @return The tensor created from the INDArray.
     * @throws IllegalArgumentException
     *         If the ndarray type is not supported.
     */
    private static <T extends Type<T>> Tensor<Integer> buildInt(RandomAccessibleInterval<IntType> ndarray)
    {
    	int[] arr = RaiArrayUtils.intArray(ndarray);
    	IntBuffer buff = IntBuffer.wrap(arr);
		Tensor<Integer> tensor = Tensor.create(ndarray.dimensionsAsLongArray(), buff);
		return tensor;
    }

    /**
     * Creates a float-typed {@link Tensor} based on the provided {@link RandomAccessibleInterval} and the desired dimension order for the resulting tensor.
     * 
     * @param ndarray
     *        The sequence to be converted.
    * @return The tensor created from the INDArray.
     * @throws IllegalArgumentException
     *         If the ndarray type is not supported.
     */
    private static <T extends Type<T>>  Tensor<Float> buildFloat(RandomAccessibleInterval<FloatType> ndarray)
    {
    	float[] arr = RaiArrayUtils.floatArray(ndarray);
    	FloatBuffer buff = FloatBuffer.wrap(arr);
		Tensor<Float> tensor = Tensor.create(ndarray.dimensionsAsLongArray(), buff);
		return tensor;
    }

    /**
     * Creates a double-typed {@link Tensor} based on the provided {@link RandomAccessibleInterval} and the desired dimension order for the resulting tensor.
     * 
     * @param ndarray
     *        The ndarray to be converted.
     * @return The tensor created from the INDArray.
     * @throws IllegalArgumentException
     *         If the ndarray type is not supported.
     */
    private static <T extends Type<T>>  Tensor<Double> buildDouble(RandomAccessibleInterval<DoubleType> ndarray)
    {
    	double[] arr = RaiArrayUtils.doubleArray(ndarray);
    	DoubleBuffer buff = DoubleBuffer.wrap(arr);
		Tensor<Double> tensor = Tensor.create(ndarray.dimensionsAsLongArray(), buff);
		return tensor;
    }
}
