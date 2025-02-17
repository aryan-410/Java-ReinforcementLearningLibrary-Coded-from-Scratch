package agent;

import A2C.Actor;
import A2C.Critic;
import environment.CartPole;
import environment.StepResult;

public class A2CAgent {
    private CartPole env;
    private Actor actor;
    private Critic critic;
    private double gamma;  // Discount factor
    private int stateDim;
    private int actionDim;

    public A2CAgent(CartPole env, double gamma, double actorLr, double criticLr) {
        this.env = env;
        this.gamma = gamma;
        // For CartPole, state dimension is 4 and there are 2 actions.
        this.stateDim = 4;
        this.actionDim = 2;
        actor = new Actor(stateDim, actionDim, actorLr);
        critic = new Critic(stateDim, criticLr);
    }

    public void train(int episodes) {
        for (int episode = 0; episode < episodes; episode++) {
            double[] state = env.reset();
            double totalReward = 0.0;
            while (!env.isDone()) {
                // Actor selects an action.
                int action = actor.selectAction(state);
                StepResult<double[]> result = env.step(action);
                double reward = result.getReward();
                double[] nextState = result.getState();
                totalReward += reward;

                // Compute TD target.
                double target = reward;
                if (!result.isDone()) {
                    target += gamma * critic.value(nextState);
                }

                // TD error (advantage): how much better (or worse) was this action?
                double value = critic.value(state);
                double advantage = target - value;

                // Update Critic (value estimator) with TD target.
                critic.update(state, target);
                // Update Actor (policy) using the advantage.
                actor.update(state, action, advantage);

                state = nextState;
            }
            System.out.println("Episode " + episode + ": Total Reward = " + totalReward);
        }
    }
}
