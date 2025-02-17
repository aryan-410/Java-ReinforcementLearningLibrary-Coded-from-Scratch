package A2C;

import java.util.Random;

public class Actor {
    private int stateDim;
    private int actionDim;
    private double[][] weights; // shape: [actionDim x stateDim]
    private double[] biases;    // shape: [actionDim]
    private double learningRate;
    private Random random;

    public Actor(int stateDim, int actionDim, double learningRate) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.learningRate = learningRate;
        this.random = new Random();

        weights = new double[actionDim][stateDim];
        biases = new double[actionDim];
        // Initialize weights to small random values
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim; j++) {
                weights[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
            biases[i] = 0.0;
        }
    }

    // Forward pass: compute logits = W * state + b, then softmax
    public double[] forward(double[] state) {
        double[] logits = new double[actionDim];
        for (int i = 0; i < actionDim; i++) {
            double sum = biases[i];
            for (int j = 0; j < stateDim; j++) {
                sum += weights[i][j] * state[j];
            }
            logits[i] = sum;
        }
        return softmax(logits);
    }

    // Softmax helper method
    private double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) {
            if (l > max) {
                max = l;
            }
        }
        double sum = 0.0;
        double[] exp = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }
        double[] probs = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = exp[i] / sum;
        }
        return probs;
    }

    // Select an action by sampling from the computed probability distribution
    public int selectAction(double[] state) {
        double[] probs = forward(state);
        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                return i;
            }
        }
        return probs.length - 1; // fallback
    }

    /**
     * Update the actor parameters using the policy gradient update.
     * For a given state and action, the gradient is proportional to:
     *    advantage * (one_hot(action) - probs)
     * This update pushes up the probability of the chosen action if advantage > 0.
     */
    public void update(double[] state, int action, double advantage) {
        double[] probs = forward(state);
        for (int i = 0; i < actionDim; i++) {
            // Compute the difference between one-hot encoding and the probability.
            double error = ((i == action) ? 1.0 : 0.0) - probs[i];
            for (int j = 0; j < stateDim; j++) {
                // Gradient ascent update: weights += lr * advantage * error * state[j]
                weights[i][j] += learningRate * advantage * error * state[j];
            }
            // Update bias: bias += lr * advantage * error
            biases[i] += learningRate * advantage * error;
        }
    }
}
