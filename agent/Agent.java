package agent;

/**
 * A generic abstract class for an RL agent.
 * S represents the state type, and A represents the action type.
 */
public abstract class Agent<S, A> {
    protected double learningRate;
    protected double discountFactor;

    public Agent(double learningRate, double discountFactor) {
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
    }

    /**
     * Given the current state, choose an action.
     * @param state The current state.
     * @return The chosen action.
     */
    public abstract A chooseAction(S state);

    /**
     * Update the agent’s knowledge based on the transition.
     * @param state The current state.
     * @param action The action taken.
     * @param reward The reward received.
     * @param nextState The resulting state.
     * @param done Whether the episode is finished.
     */
    public abstract void learn(S state, A action, double reward, S nextState, boolean done);
}
