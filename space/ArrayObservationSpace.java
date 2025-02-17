package space;

public class ArrayObservationSpace {
    private final int[] shape;
    private final double[] low;
    private final double[] high;

    public ArrayObservationSpace(int[] shape, double[] low, double[] high) {
        this.shape = shape;
        this.low = low;
        this.high = high;
    }

    public double[] getHigh() {
        return high;
    }

    public double[] getLow() {
        return low;
    }

    public int[] getShape() {
        return shape;
    }
}
