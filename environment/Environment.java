package environment;

import space.State;
import space.Action;

/**
 * An abstract class representing a reinforcement learning environment.
 * @param <S> the type representing the state.
 * @param <A> the type representing the action.
 */
public abstract class Environment<S extends State, A> {

    /**
     * Resets the environment to its initial state.
     */


    public abstract S reset();

    /**
     * Executes the action returned by the model
     */
    public abstract void execute(A action);

    /**
     * Applies an action to the environment and returns the resulting transition.
     * @param action the action to take.
     * @return a StepResult containing the next state, reward, and done flag.
     */
    public abstract StepResult<S> step(A action);

    /**
     * Checks if the environment has reached a terminal state.
     * @return true if the episode is over; false otherwise.
     */
    public abstract boolean isDone();
}
