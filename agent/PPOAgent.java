package agent;

import A2C.PPOActor;
import A2C.PPOCritic;
import environment.CartPole;
import environment.StepResult;
import util.Transition;

import java.util.ArrayList;
import java.util.List;

public class PPOAgent {
    private CartPole env;
    private PPOActor actor;
    private PPOCritic critic;
    private double gamma;
    private int stateDim;
    private int actionDim;

    // PPO hyperparameters:
    private double clipEpsilon;
    private int updateEpochs; // number of passes over the collected trajectory
    // For simplicity, we use the entire episode as one batch.

    public PPOAgent(CartPole env, double gamma, double actorLr, double criticLr,
                    double clipEpsilon, int updateEpochs) {
        this.env = env;
        this.gamma = gamma;
        // For CartPole, state dimension is 4 and there are 2 actions.
        this.stateDim = 4;
        this.actionDim = 2;
        this.actor = new PPOActor(stateDim, actionDim, actorLr);
        this.critic = new PPOCritic(stateDim, criticLr);
        this.clipEpsilon = clipEpsilon;
        this.updateEpochs = updateEpochs;
    }

    public void train(int episodes) {
        List<Transition> trajectory = new ArrayList<>();

        for (int episode = 0; episode < episodes; episode++) {
            double[] state = env.reset();
            double totalReward = 0.0;
            trajectory.clear();

            // Collect one episode.
            while (!env.isDone()) {
                Transition t = new Transition();
                t.state = state.clone();
                // Actor selects an action and stores its log probability in the transition.
                int action = actor.selectAction(state, t);
                t.action = action;

                // Get critic’s value estimate.
                t.value = critic.value(state);

                // Take the action in the environment.
                StepResult<double[]> result = env.step(action);
                t.reward = result.getReward();
                totalReward += t.reward;

                state = result.getState().clone();

                trajectory.add(t);
            }

            // Compute discounted returns and advantages for the episode.
            double G = 0;
            for (int i = trajectory.size() - 1; i >= 0; i--) {
                Transition t = trajectory.get(i);
                G = t.reward + gamma * G;
                t.returnG = G;
                t.advantage = t.returnG - t.value;
            }

            // Perform several epochs of update over the collected trajectory.
            for (int epoch = 0; epoch < updateEpochs; epoch++) {
                actor.updateBatch(trajectory, clipEpsilon);
                critic.updateBatch(trajectory);
            }

            System.out.println("Episode " + episode + ": Total Reward = " + totalReward);
        }
    }
}
