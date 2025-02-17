package space;

public class ContinousState extends State {
    private double[] current;
    private double[] minBounds;
    private double[] maxBounds;
    private int features;
    private int bins;

    public ContinousState(int features, double[] initialState, double[] minBounds, double[] maxBounds, int bins) {
        if (features != initialState.length) {
            throw new IllegalArgumentException("Error: the number of features doesn't match the length of features ");
        }

        this.features = features;
        this.current = initialState;
        this.bins = bins;

        this.minBounds = minBounds;
        this.maxBounds = maxBounds;
    }

    private int discretize(double[] state) {
        int discreteValue = 0;
        for (int i = 0; i < features; i++) {
            int digit = discretizeValue(current[i], minBounds[i], maxBounds[i]);
            discreteValue = discreteValue * bins + digit;
        }

        return discreteValue;
    }

    public ContinousState(int features, double[] initialState, double[] minBounds, double[] maxBounds) {
        this(features, initialState, minBounds, maxBounds, 15);
    }

    private int discretizeValue(double value, double min, double max) {
        value = Math.max(Math.min(value, max), min);

        double binSize = (max - min) / bins;
        int bin = (int) ((value - min) / binSize);
        bin = Math.min(bin, bins);
        return bin;
    }

    @Override
    public State copy() {
        return null;
    }

    @Override
    public boolean isTerminal() {
        return false;
    }
}
