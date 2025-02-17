package space;

/**
 * An abstract representation of an action in RL.
 * Concrete classes (e.g., DiscreteAction, ContinuousAction) will extend this.
 */
public interface ActionSpace<A> {
    default boolean isValid(A action) {return true;}
    int[] getShape();
    A randomAction();
}
