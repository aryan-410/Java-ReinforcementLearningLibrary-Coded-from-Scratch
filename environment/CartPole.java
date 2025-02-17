package environment;

import space.ContinousState;

import java.util.Random;

/**
 * A simple simulation of the CartPole environment.
 * The state is represented as a double array:
 * [cart position, cart velocity, pole angle, pole angular velocity].
 * The action is an Integer: 0 for left force, 1 for right force.
 */
public class CartPole extends Environment<ContinousState, Integer> {

    @Override
    public void execute(Integer action) {

    }

    // Physical constants
    private static final double GRAVITY = 9.8;
    private static final double MASS_CART = 1.0;
    private static final double MASS_POLE = 0.1;
    private static final double TOTAL_MASS = MASS_CART + MASS_POLE;
    private static final double LENGTH = 0.5; // half the pole length
    private static final double POLEMASS_LENGTH = MASS_POLE * LENGTH;
    private static final double FORCE_MAG = 10.0;
    private static final double TAU = 0.02; // time step (seconds)

    // Termination thresholds
    private static final double X_THRESHOLD = 2.4;
    private static final double THETA_THRESHOLD_RADIANS = 15 * Math.PI / 180;

    private double[] state;
    private boolean done;
    private Random random;

    public CartPole() {
        random = new Random();
        reset();
    }

    @Override
    public double[] reset() {
        state = new double[4];
        // Initialize state with small random values near 0.
        state[0] = random.nextDouble() * 0.08 - 0.04; // cart position
        state[1] = random.nextDouble() * 0.08 - 0.04; // cart velocity
        state[2] = random.nextDouble() * 0.08 - 0.04; // pole angle
        state[3] = random.nextDouble() * 0.08 - 0.04; // pole angular velocity
        done = false;
        return state.clone();
    }

    @Override
    public StepResult<double[]> step(Integer action) {
        if (done) {
            return new StepResult<>(state, 0, true);
        }

        // Interpret the action: 0 = left force, 1 = right force.
        double force = (action == 1) ? FORCE_MAG : -FORCE_MAG;
        double x = state[0];
        double x_dot = state[1];
        double theta = state[2];
        double theta_dot = state[3];

        double costheta = Math.cos(theta);
        double sintheta = Math.sin(theta);

        // Compute the accelerations using the dynamics equations.
        double temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;
        double thetaacc = (GRAVITY * sintheta - costheta * temp)
                / (LENGTH * (4.0/3.0 - MASS_POLE * costheta * costheta / TOTAL_MASS));
        double xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

        // Update state using Euler integration.
        x += TAU * x_dot;
        x_dot += TAU * xacc;
        theta += TAU * theta_dot;
        theta_dot += TAU * thetaacc;

        state[0] = x;
        state[1] = x_dot;
        state[2] = theta;
        state[3] = theta_dot;

        // Check if the state is beyond the thresholds.
        done = (x < -X_THRESHOLD || x > X_THRESHOLD ||
                theta < -THETA_THRESHOLD_RADIANS || theta > THETA_THRESHOLD_RADIANS);

        // Reward of 1 for each time step until termination.
        double reward = done ? 0.0 : 1.0;

        return new StepResult<>(state.clone(), reward, done);
    }

    @Override
    public boolean isDone() {
        return done;
    }
}
