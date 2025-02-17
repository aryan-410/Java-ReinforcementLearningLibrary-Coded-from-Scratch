package util;

public class Transition {
    public double[] state;   // state at time t
    public int action;       // action taken
    public double reward;    // immediate reward
    public double oldLogProb; // log probability under the old policy
    public double value;     // critic's value estimate at state
    public double returnG;   // discounted return computed later
    public double advantage; // advantage: returnG - value
}
