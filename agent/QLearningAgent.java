package agent;

import policy.Policy;
import java.util.HashMap;
import java.util.Map;

/**
 * A Q-learning agent that discretizes the continuous CartPole state.
 * Uses a Q-table stored as a map from discretized state keys to arrays of Q-values.
 */
public class QLearningAgent extends Agent<double[], Integer> {
    private Map<String, double[]> qTable;
    private int numActions;
    private Policy<double[], Integer> policy;
    private int bins = 20; // number of bins for discretization

    public QLearningAgent(double learningRate, double discountFactor, Policy<double[], Integer> policy, int numActions) {
        super(learningRate, discountFactor);
        this.policy = policy;
        this.numActions = numActions;
        qTable = new HashMap<>();
    }

    /**
     * Discretizes the continuous state into a string key.
     * Assumes state = [x, x_dot, theta, theta_dot].
     */
    private String discretize(double[] state) {
        // Define assumed ranges for each variable.
        double xMin = -2.4, xMax = 2.4;
        double xDotMin = -3.0, xDotMax = 3.0;
        double thetaMin = -0.209, thetaMax = 0.209; // approximately ±12 degrees (radians)
        double thetaDotMin = -3.5, thetaDotMax = 3.5;

        int xBin = discretizeValue(state[0], xMin, xMax, bins);
        int xDotBin = discretizeValue(state[1], xDotMin, xDotMax, bins);
        int thetaBin = discretizeValue(state[2], thetaMin, thetaMax, bins);
        int thetaDotBin = discretizeValue(state[3], thetaDotMin, thetaDotMax, bins);

        return xBin + "_" + xDotBin + "_" + thetaBin + "_" + thetaDotBin;
    }

    /**
     * Helper method to discretize a single value.
     */
    private int discretizeValue(double value, double min, double max, int bins) {
        value = Math.max(Math.min(value, max), min);

        double binSize = (max - min) / bins;
        int bin = (int) ((value - min) / binSize);
        bin = Math.min(bin, bins);
        return bin;
    }

    @Override
    public Integer chooseAction(double[] state) {
        String key = discretize(state);
        if (!qTable.containsKey(key)) {
            qTable.put(key, new double[numActions]);
        }
        double[] qValues = qTable.get(key);
        return policy.chooseAction(state, qValues);
    }

    @Override
    public void learn(double[] state, Integer action, double reward, double[] nextState, boolean done) {
        String key = discretize(state);
        if (!qTable.containsKey(key)) {
            qTable.put(key, new double[numActions]);
        }
        String nextKey = discretize(nextState);
        if (!qTable.containsKey(nextKey)) {
            qTable.put(nextKey, new double[numActions]);
        }
        double[] qValues = qTable.get(key);
        double[] nextQValues = qTable.get(nextKey);

        // Compute the maximum Q value for the next state.
        double maxNextQ = 0.0;
        if (!done) {
            maxNextQ = nextQValues[0];
            for (int i = 1; i < numActions; i++) {
                if (nextQValues[i] > maxNextQ) {
                    maxNextQ = nextQValues[i];
                }
            }
        }
        // Q-learning update rule.
        qValues[action] = qValues[action] + learningRate * (reward + discountFactor * maxNextQ - qValues[action]);
    }
}
