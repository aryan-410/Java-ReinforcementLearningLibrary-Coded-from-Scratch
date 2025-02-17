package space;

public abstract class DiscreteState extends State {
    private int bins;

    public DiscreteState(int features, double[][] bounds) { this(features, bounds, 15);}

    public DiscreteState(int features, double[][] bounds, int bins) {
        this.bins = 15;
    }
}