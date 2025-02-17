package A2C;

import util.Transition;

import java.util.List;
import java.util.Random;

public class PPOActor {
    private int stateDim;
    private int actionDim;
    private double[][] weights; // shape: [actionDim x stateDim]
    private double[] biases;    // shape: [actionDim]
    private double learningRate;
    private Random random;

    public PPOActor(int stateDim, int actionDim, double learningRate) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.learningRate = learningRate;
        this.random = new Random();

        weights = new double[actionDim][stateDim];
        biases = new double[actionDim];
        // Initialize weights with small random values.
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim; j++) {
                weights[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
            biases[i] = 0.0;
        }
    }

    // Compute logits = W * state + b, then apply softmax to get action probabilities.
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

    /**
     * Selects an action based on the current policy.
     * Also stores the old log probability in the provided Transition.
     */
    public int selectAction(double[] state, Transition transition) {
        double[] probs = forward(state);
        double r = random.nextDouble();
        double cumulative = 0.0;
        int action = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                action = i;
                break;
            }
        }
        // Store the log probability for PPO's ratio calculation later.
        transition.oldLogProb = Math.log(probs[action]);
        return action;
    }

    /**
     * Update the actor's parameters using the PPO clipped surrogate objective.
     * For each transition in the batch:
     *   - Compute the new log probability
     *   - Compute the probability ratio: r = exp(newLogProb - oldLogProb)
     *   - Form the surrogate: min(r * advantage, clip(r, 1-ε, 1+ε)*advantage)
     *   - Compute the gradient of log(prob) and update weights.
     */
    public void updateBatch(List<Transition> batch, double clipEpsilon) {
        // Loop over each transition in the batch.
        for (Transition t : batch) {
            double[] probs = forward(t.state);
            double newLogProb = Math.log(probs[t.action]);
            double ratio = Math.exp(newLogProb - t.oldLogProb);
            double unclipped = ratio * t.advantage;
            double clippedRatio = Math.min(Math.max(ratio, 1 - clipEpsilon), 1 + clipEpsilon);
            double clipped = clippedRatio * t.advantage;
            // Use the minimum of the two surrogate objectives.
            double surrogate = Math.min(unclipped, clipped);

            // For a softmax policy, the gradient of log probability for the chosen action:
            //   ∇ log π(a|s) = one_hot(a) - probs.
            // We update each weight as: Δ = learningRate * surrogate * (indicator - prob).
            for (int i = 0; i < actionDim; i++) {
                double indicator = (i == t.action) ? 1.0 : 0.0;
                double gradCoefficient = surrogate * (indicator - probs[i]);
                for (int j = 0; j < stateDim; j++) {
                    weights[i][j] += learningRate * gradCoefficient * t.state[j];
                }
                biases[i] += learningRate * gradCoefficient;
            }
        }
    }
}
