package policy;

/**
 * An abstract class for decision policies.
 * @param <S> the type representing the state.
 * @param <A> the type representing the action.
 */
public abstract class Policy<S, A> {

    /**
     * Given a state and an array of Q-values for the available actions,
     * selects and returns an action.
     *
     * @param state the current state.
     * @param qValues an array of Q-values corresponding to possible actions.
     * @return the selected action.
     */
    public abstract A chooseAction(S state, double[] qValues);
}
