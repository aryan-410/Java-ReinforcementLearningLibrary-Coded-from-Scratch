package examples.CartPole;

import agent.QLearningAgent;
import environment.CartPole;
import environment.StepResult;
import policy.EpsilonGreedyPolicy;

/**
 * A runnable example that trains a Q-learning agent to balance the CartPole using an epsilon-greedy policy.
 */
public class CartPoleExample {
    public static void main(String[] args) {

        int episodes = 1000;
        int maxSteps = 200;

        // Create the CartPole environment.
        CartPole env = new CartPole();

        // Create an epsilon-greedy policy with epsilon = 0.1.
        EpsilonGreedyPolicy<double[], Integer> policy = new EpsilonGreedyPolicy<>(1/(episodes * 0.20 * maxSteps));

        // Create a Q-learning agent with learning rate 0.1, discount factor 0.99, and 2 actions (left/right).
        QLearningAgent agent = new QLearningAgent(0.1, 0.99, policy, 2);

        for (int episode = 1; episode <= episodes; episode++) {
            double[] state = env.reset();
            double totalReward = 0;

            for (int step = 0; step < maxSteps; step++) {
                // Agent selects an action based on the current state.
                int action = agent.chooseAction(state);

                // The environment processes the action.
                StepResult<double[]> result = env.step(action);
                double reward = result.getReward();
                double[] nextState = result.getNextState();
                boolean done = result.isDone();

                // Agent learns from the experience.
                agent.learn(state, action, reward, nextState, done);

                state = nextState;
                totalReward += reward;

                if (done) {
                    break;
                }
            }
            System.out.println("Episode " + episode + " finished with reward: " + totalReward);
        }
    }
}
