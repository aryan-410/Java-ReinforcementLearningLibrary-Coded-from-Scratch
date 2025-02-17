package examples.CartPole;

import agent.QLearningAgent;
import environment.CartPole;
import environment.StepResult;
import policy.EpsilonGreedyPolicy;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.concurrent.CountDownLatch;

/**
 * Runs a training loop for a Q-learning agent on the CartPole environment.
 * It prints the average reward every 100 episodes.
 * Every 100 episodes it also runs one demonstration episode that is visualized.
 */
public class CartPoleExampleWithVisualization {

    public static void main(String[] args) {
        int totalEpisodes = 2000;
        int maxSteps = 500;
        int batchSize = 100;
        double batchRewardSum = 0.0;

        // Create the environment and agent.
        CartPole env = new CartPole();
        // Create an epsilon-greedy policy with epsilon = 0.1.
        EpsilonGreedyPolicy<double[], Integer> policy = new EpsilonGreedyPolicy<>(1/(totalEpisodes * 0.30 * maxSteps));
        // Create a Q-learning agent with learning rate 0.1, discount factor 0.99, and 2 actions.
        QLearningAgent agent = new QLearningAgent(0.1, 0.99, policy, 2);


        for (int episode = 1; episode <= totalEpisodes; episode++) {
            // Reset the environment.
            double[] state = env.reset();
            double episodeReward = 0.0;

            // Run one training episode.
            for (int step = 0; step < maxSteps; step++) {
                int action = agent.chooseAction(state);
                StepResult<double[]> result = env.step(action);
                double reward = result.getReward();
                double[] nextState = result.getNextState();
                boolean done = result.isDone();

                agent.learn(state, action, reward, nextState, done);

                state = nextState;
                episodeReward += reward;

                if (done) {
                    break;
                }
            }

            batchRewardSum += episodeReward;

            // Every batchSize episodes, print the average reward and run a demonstration.
            if (episode % batchSize == 0) {
                double avgReward = batchRewardSum / batchSize;
                System.out.println("Episodes " + (episode - batchSize + 1) + " to " + episode + " average reward: " + avgReward);
                batchRewardSum = 0.0;

                // Run a demonstration episode with visualization.
                runVisualizationEpisode(agent, episode);
            }
        }
    }

    /**
     * Runs one demonstration episode in which the agent interacts with a new CartPole
     * environment and the results are visualized using a Swing JFrame.
     * @param agent the Q-learning agent.
     * @param episodeNumber the current episode number (to be displayed).
     */
    private static JFrame demoFrame = null;
    private static CartPoleVisualizer demoVisualizer = null;

    private static void runVisualizationEpisode(QLearningAgent agent, int episodeNumber) {
        // Create a new environment for demonstration.
        CartPole demoEnv = new CartPole();
        // Wrap the state in a one-element array so it can be updated from within the inner class.
        final double[][] stateHolder = new double[][]{ demoEnv.reset() };

        // Reuse the existing demoFrame and demoVisualizer if available, otherwise create them.
        if (demoFrame == null) {
            demoFrame = new JFrame("CartPole Demonstration");
            demoVisualizer = new CartPoleVisualizer();
            demoFrame.getContentPane().add(demoVisualizer);
            demoFrame.pack();
            demoFrame.setLocationRelativeTo(null);
            demoFrame.setVisible(true);
        }

        // Update the visualizer with the current episode number.
        demoVisualizer.setEpisodeNumber(episodeNumber);

        // Create a latch to block the training thread until the demonstration is complete.
        CountDownLatch latch = new CountDownLatch(1);

        // Use a Swing Timer to run the demonstration simulation.
        int delay = 20; // ~50 frames per second
        Timer timer = new Timer(delay, null);
        timer.addActionListener(new ActionListener() {
            int step = 0;
            final int maxDemoSteps = 200;

            @Override
            public void actionPerformed(ActionEvent e) {
                if (demoEnv.isDone() || step >= maxDemoSteps) {
                    timer.stop();
                    latch.countDown();
                } else {
                    // Let the agent choose an action for the current state.
                    int action = agent.chooseAction(stateHolder[0]);
                    StepResult<double[]> result = demoEnv.step(action);
                    stateHolder[0] = result.getNextState();
                    // Update the visualizer with the current cart position (index 0) and pole angle (index 2).
                    demoVisualizer.updateState(stateHolder[0][0], stateHolder[0][2]);
                    step++;
                }
            }
        });
        timer.start();

        // Block until the demonstration simulation is complete.
        try {
            latch.await();
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
    }
}
