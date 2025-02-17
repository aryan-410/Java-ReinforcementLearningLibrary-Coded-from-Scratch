package A2C;

import java.util.Random;

public class Critic {
    private int stateDim;
    private double[] weights; // shape: [stateDim]
    private double bias;
    private double learningRate;
    private Random random;

    public Critic(int stateDim, double learningRate) {
        this.stateDim = stateDim;
        this.learningRate = learningRate;
        this.random = new Random();

        weights = new double[stateDim];
        for (int i = 0; i < stateDim; i++) {
            weights[i] = (random.nextDouble() - 0.5) * 0.1;
        }
        bias = 0.0;
    }

    // Compute value: v = w^T * state + b
    public double value(double[] state) {
        double v = bias;
        for (int i = 0; i < stateDim; i++) {
            v += weights[i] * state[i];
        }
        return v;
    }

    // Update the critic using the TD error: error = target - value(state)
    public void update(double[] state, double target) {
        double v = value(state);
        double error = target - v;
        for (int i = 0; i < stateDim; i++) {
            weights[i] += learningRate * error * state[i];
        }
        bias += learningRate * error;
    }
}
