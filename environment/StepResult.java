package environment;

import jdk.internal.org.objectweb.asm.tree.AbstractInsnNode;

public class StepResult<S> {
    private final S nextState;
    private final double reward;
    private final boolean done;

    public StepResult(S nextState, double reward, boolean done) {
        this.nextState = nextState;
        this.reward = reward;
        this.done = done;
    }

    public S getNextState() {
        return nextState;
    }

    public double getReward() {
        return reward;
    }

    public boolean isDone() {
        return done;
    }

    public AbstractInsnNode getState() {
        return null;
    }
}
