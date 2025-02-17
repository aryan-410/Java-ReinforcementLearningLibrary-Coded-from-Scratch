package A2C;

import util.Transition;

import java.util.List;
import java.util.Random;

public class PPOCritic {
    private int stateDim;
    private double[] weights; // shape: [stateDim]
    private double bias;
    private double learningRate;
    private Random random;

    public PPOCritic(int stateDim, double learningRate) {
        this.stateDim = stateDim;
        this.learningRate = learningRate;
        this.random = new Random();
        weights = new double[stateDim];
        for (int i = 0; i < stateDim; i++) {
            weights[i] = (random.nextDouble() - 0.5) * 0.1;
        }
        bias = 0.0;
    }

    // Compute the state value: v = w^T * state + b.
    public double value(double[] state) {
        double v = bias;
        for (int i = 0; i < stateDim; i++) {
            v += weights[i] * state[i];
        }
        return v;
    }

    // Update the critic by performing a gradient step on the value loss (squared error).
    public void updateBatch(List<Transition> batch) {
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
