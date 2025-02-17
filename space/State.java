package space;

import java.util.Objects;

/**
 * An abstract representation of a state in RL.
 * Concrete classes (e.g., GridWorldState, AtariState) will extend this.
 */
public abstract class State {
    /**
     * Often, states need to be copied (e.g., for planning, or to store safely
     * in replay buffers without mutating the original). Subclasses can override
     * with environment-specific logic.
     */
    public abstract State copy();

    /**
     * Whether this state is terminal (end of episode).
     * Subclasses can override with logic to check if the game/environment is done.
     */
    public abstract boolean isTerminal();

    /**
     * Optionally, define a way to convert the state to an array
     * (common for feeding into neural networks).
     * Subclasses can provide the specifics.
     */
    public double[] toArray() {
        // Default implementation can be empty or throw an exception.
        // Subclasses override if needed.
        throw new UnsupportedOperationException("toArray() not implemented");
    }

    /**
     * Subclasses can override equals and hashCode if you plan to use states as map keys
     * or want to compare them for uniqueness. Provide a default or let children define it.
     */
    @Override
    public boolean equals(Object o) {
        return super.equals(o);
    }

    @Override
    public int hashCode() {
        return Objects.hash();
    }

    @Override
    public String toString() {
        return "State{}";
    }
}