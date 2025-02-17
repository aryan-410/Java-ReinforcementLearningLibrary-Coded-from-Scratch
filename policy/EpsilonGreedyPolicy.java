package policy;

import java.util.Random;

/**
 * Implements an epsilon-greedy strategy.
 * With probability ε, a random action is chosen;
 * otherwise, the action with the highest Q-value is selected.
 */
public class EpsilonGreedyPolicy<S, A> extends Policy<S, A> {
    private double epsilon;
    private Random random;
    private double increament;

    public EpsilonGreedyPolicy(double increament) {
        this.epsilon = 0.1;
        this.random = new Random();
        this.increament = increament;
    }

    @Override
    public A chooseAction(S state, double[] qValues) {
        A bestAction;
        if (random.nextDouble() < epsilon) {
            // Exploration: choose a random action.
            int numActions = qValues.length;
            // Here we assume that A is Integer.
            bestAction = (A) Integer.valueOf(random.nextInt(numActions));
        } else {
            // Exploitation: choose the best action.
            int bestActionInt = 0;
            for (int i = 1; i < qValues.length; i++) {
                if (qValues[i] > qValues[bestActionInt]) {
                    bestActionInt = i;
                }
            }
            bestAction = (A) Integer.valueOf(bestActionInt);
        }

        epsilon += increament;
        return bestAction;
    }
}
