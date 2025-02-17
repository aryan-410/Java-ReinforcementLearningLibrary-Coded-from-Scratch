package trainer;

import agent.Agent;
import environment.Environment;

/**
 * An abstract class for training an RL agent.
 * @param <S> the type representing the state.
 * @param <A> the type representing the action.
 */
public abstract class Trainer<S, A> {
    protected Agent<S, A> agent;
    protected Environment<S, A> environment;

    /**
     * Constructs a Trainer with the specified agent and environment.
     * @param agent the RL agent.
     * @param environment the environment in which the agent will be trained.
     */
    public Trainer(Agent<S, A> agent, Environment<S, A> environment) {
        this.agent = agent;
        this.environment = environment;
    }

    /**
     * Runs the training process for a specified number of episodes.
     * @param episodes the number of episodes to train for.
     */
    public abstract void train(int episodes);
}
