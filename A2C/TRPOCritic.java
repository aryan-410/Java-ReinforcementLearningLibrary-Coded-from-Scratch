package A2C;

import java.util.Random;

public class TRPOCritic {
    private int stateDim;
    private double[] weights;
    private double bias;
    private double learningRate;
    private Random random;

    public TRPOCritic(int stateDim, double learningRate) {
        this.stateDim = stateDim;
        this.learningRate = learningRate;
        random = new Random();
        weights = new double[stateDim];
        for (int i = 0; i < stateDim; i++) {
            weights[i] = (random.nextDouble() - 0.5) * 0.1;
        }
        bias = 0.0;
    }

    // Compute the state value: V(s) = w^T * state + b.
    public double value(double[] state) {
        double v = bias;
        for (int i = 0; i < stateDim; i++) {
            v += weights[i] * state[i];
        }
        return v;
    }

    // Update using a simple gradient step on the squared error.
    public void update(double[] state, double target) {
        double v = value(state);
        double error = target - v;
        for (int i = 0; i < stateDim; i++) {
            weights[i] += learningRate * error * state[i];
        }
        bias += learningRate * error;
    }

    // Batch update over a set of transitions.
    public void updateBatch(java.util.List<Transition> batch) {
        for (Transition t : batch) {
            double v = value(t.state);
            double error = t.returnG - v;
            for (int i = 0; i < stateDim; i++) {
                weights[i] += learningRate * error * t.state[i];
            }
            bias += learningRate * error;
        }
    }
}
