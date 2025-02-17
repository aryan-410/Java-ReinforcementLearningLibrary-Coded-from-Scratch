package main;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Random;

/**
 * A single-file implementation of CartPole with a Q-learning agent that uses
 * an epsilon-greedy policy. Includes a Swing GUI for visualization.
 *
 * The hyperparameters have been tuned so that the agent actually learns:
 *  - Higher learning rate
 *  - Larger epsilon decay
 *  - More bins
 *  - Rewards: +1 per step alive, -100 on failure
 */
public class CartPoleEpsilonGreedy extends JPanel {

    //---------------------------
    // Environment hyperparameters
    //---------------------------
    private static final double GRAVITY = 9.8;
    private static final double CART_MASS = 1.0;
    private static final double POLE_MASS = 0.1;
    private static final double TOTAL_MASS = (CART_MASS + POLE_MASS);
    private static final double POLE_LENGTH = 0.5; // half the pole length
    private static final double FORCE_MAG = 10.0;
    private static final double TAU = 0.02; // seconds per step (20ms)
    private static final double FOUR_THIRDS = 4.0 / 3.0;

    // Termination conditions
    private static final double THETA_THRESHOLD_RADIANS = 24 * Math.PI / 180; // ~12 deg
    private static final double X_THRESHOLD = 2.4; // cart fails if |x| > 2.4

    // Render scaling
    private static final int SCALE = 200;   // px per meter
    private static final int CART_WIDTH = 40;
    private static final int CART_HEIGHT = 20;
    private static final int POLE_WIDTH = 6;
    private static final int POLE_HEIGHT = (int)(POLE_LENGTH * 2 * SCALE);

    //------------------------------------------
    // Q-Learning and discretization parameters
    //------------------------------------------
    // Binning: More bins => finer resolution => better (but bigger Q-table).
    private static final int NUM_BINS_CART_POS = 25;
    private static final int NUM_BINS_CART_VEL = 25;
    private static final int NUM_BINS_POLE_ANGLE = 25;
    private static final int NUM_BINS_POLE_ANG_VEL = 25;
    private static final int NUM_ACTIONS = 2; // 0=left, 1=right

    // Learning hyperparameters
    private static final double LEARNING_RATE = 0.02;
    private static final double DISCOUNT_FACTOR = 0.99;
    // Epsilon starts high so we explore a lot initially
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_MIN = 0.01;
    // Decay fairly slowly so it has time to explore
    private static final double EPSILON_DECAY = 0.999995;

    private static final int MAX_EPISODES = 50000;
    private static final int MAX_STEPS_PER_EPISODE = 5000;
    private static final int RENDER_FREQUENCY = 200; // how often to do a test render

    //----------------------
    // Environment state
    //----------------------
    private double cartPos = 0.0;       // x
    private double cartVel = 0.0;       // x_dot
    private double poleAngle = 0.0;     // theta
    private double poleAngleVel = 0.0;  // theta_dot

    // Q-Table: shape [numStates][numActions]
    // Flatten 4D state bins into 1D index
    private static final int NUM_STATES =
            NUM_BINS_CART_POS *
                    NUM_BINS_CART_VEL *
                    NUM_BINS_POLE_ANGLE *
                    NUM_BINS_POLE_ANG_VEL;

    // Each entry: QTable[state][action]
    private double[][] QTable = new double[NUM_STATES][NUM_ACTIONS];

    // For the GUI
    private JFrame frame;
    private int renderEpisode = 0; // which episode we show in the GUI

    // RNG
    private Random rand = new Random(42);

    public CartPoleEpsilonGreedy() {
        // Swing window
        frame = new JFrame("CartPole Q-Learning (Single File)");
        frame.setSize(800, 400);
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

        // On close, exit
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                frame.dispose();
                System.exit(0);
            }
        });

        frame.add(this);
        frame.setVisible(true);
    }

    // Convert continuous state to discrete bin indices, then flatten to single index
    private int getStateIndex(double x, double xDot, double theta, double thetaDot) {
        // The ranges below must be somewhat large so we don't constantly clip
        // (e.g., cartVel might exceed ±2 if we push for a while)
        int cposBin = discretize(x, -2.4, 2.4, NUM_BINS_CART_POS);
        int cvelBin = discretize(xDot, -3.0, 3.0, NUM_BINS_CART_VEL);
        int pangBin = discretize(theta, -THETA_THRESHOLD_RADIANS, THETA_THRESHOLD_RADIANS, NUM_BINS_POLE_ANGLE);
        int pangvBin = discretize(thetaDot, -4.0, 4.0, NUM_BINS_POLE_ANG_VEL);

        int idx = cposBin;
        idx = idx * NUM_BINS_CART_VEL + cvelBin;
        idx = idx * NUM_BINS_POLE_ANGLE + pangBin;
        idx = idx * NUM_BINS_POLE_ANG_VEL + pangvBin;
        return idx;
    }

    // Convert a continuous value into [0..nBins-1]
    private int discretize(double value, double minValue, double maxValue, int nBins) {
        // clip
        if (value < minValue) value = minValue;
        if (value > maxValue) value = maxValue;
        double ratio = (value - minValue) / (maxValue - minValue);
        return (int) Math.floor(ratio * (nBins - 1));
    }

    // Reset environment at start of each episode
    private void resetEnvironment() {
        cartPos = 0.0;
        cartVel = 0.0;
        // small random angle so it's not always 0
        poleAngle = (rand.nextDouble() - 0.5) * 0.2;
        poleAngleVel = 0.0;
    }

    /**
     * Step the environment with the given action (0=left, 1=right).
     * Returns { reward, doneFlag }, with doneFlag=1.0 if terminal.
     */
    private double[] step(int action) {
        double force = (action == 1) ? FORCE_MAG : -FORCE_MAG;

        double cosTheta = Math.cos(poleAngle);
        double sinTheta = Math.sin(poleAngle);

        // Common intermediate terms
        double temp = (force + POLE_MASS * POLE_LENGTH * poleAngleVel * poleAngleVel * sinTheta) / TOTAL_MASS;
        double thetaAcc = (GRAVITY * sinTheta - cosTheta * temp)
                / (POLE_LENGTH * (FOUR_THIRDS - POLE_MASS * cosTheta * cosTheta / TOTAL_MASS));
        double xAcc = temp - (POLE_MASS * POLE_LENGTH * thetaAcc * cosTheta) / TOTAL_MASS;

        // Euler update
        cartPos += TAU * cartVel;
        cartVel += TAU * xAcc;
        poleAngle += TAU * poleAngleVel;
        poleAngleVel += TAU * thetaAcc;

        // Check if done
        boolean done = false;
        double reward = 1.0; // +1 for surviving each step

        // If out of bounds, mark done
        if (cartPos < -X_THRESHOLD || cartPos > X_THRESHOLD ||
                poleAngle < -THETA_THRESHOLD_RADIANS || poleAngle > THETA_THRESHOLD_RADIANS) {
            done = true;
            // Big negative reward for failing
            reward = -100.0;
        }

        return new double[]{reward, done ? 1.0 : 0.0};
    }

    // Epsilon-greedy action selection
    private int chooseAction(int stateIndex, double epsilon) {
        if (rand.nextDouble() < epsilon) {
            return rand.nextInt(NUM_ACTIONS); // explore
        } else {
            // exploit
            double[] qVals = QTable[stateIndex];
            return (qVals[0] > qVals[1]) ? 0 : 1;
        }
    }

    // Main training loop
    public void train() {
        double epsilon = EPSILON_START;

        for (int episode = 1; episode <= MAX_EPISODES; episode++) {
            resetEnvironment();
            int stateIdx = getStateIndex(cartPos, cartVel, poleAngle, poleAngleVel);

            double totalReward = 0.0;
            boolean done = false;

            for (int t = 0; t < MAX_STEPS_PER_EPISODE; t++) {
                // Pick action
                int action = chooseAction(stateIdx, epsilon);

                // Env step
                double[] stepOut = step(action);
                double reward = stepOut[0];
                done = (stepOut[1] > 0.5);

                int nextStateIdx = getStateIndex(cartPos, cartVel, poleAngle, poleAngleVel);

                // Update Q
                double bestNextQ = Math.max(QTable[nextStateIdx][0], QTable[nextStateIdx][1]);
                double tdTarget = reward + DISCOUNT_FACTOR * bestNextQ;
                double tdError = tdTarget - QTable[stateIdx][action];
                QTable[stateIdx][action] += LEARNING_RATE * tdError;

                totalReward += reward;
                stateIdx = nextStateIdx;

                // Decay epsilon
                epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);

                if (done) {
                    break;
                }
            }

            // Print progress
            System.out.printf("Episode %4d | Steps = %-3d | Total Reward = %6.2f | Eps = %.3f\n",
                    episode, (int) Math.min(MAX_STEPS_PER_EPISODE, totalReward + 100)/1, totalReward, epsilon);

            // Occasionally test-run with rendering
            if (episode % RENDER_FREQUENCY == 0) {
                renderEpisode = episode;
                testRun(); // no learning, just to show behavior
            }
        }
    }

    // A single run (with epsilon=0) to visualize learned policy
    private void testRun() {
        resetEnvironment();
        int stateIdx = getStateIndex(cartPos, cartVel, poleAngle, poleAngleVel);

        for (int t = 0; t < MAX_STEPS_PER_EPISODE; t++) {
            // Exploit only
            double[] qVals = QTable[stateIdx];
            int action = (qVals[0] > qVals[1]) ? 0 : 1;

            double[] stepOut = step(action);
            boolean done = (stepOut[1] > 0.5);

            stateIdx = getStateIndex(cartPos, cartVel, poleAngle, poleAngleVel);

            repaint(); // update GUI
            try {
                Thread.sleep(20); // 20ms per step
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if (done) break;
        }
    }

    //--------------------------------
    // Rendering (Swing)
    //--------------------------------
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        int w = getWidth();
        int h = getHeight();
        int originX = w / 2;
        int originY = h / 2 + 50;

        // white background
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, w, h);

        // ground line
        g.setColor(Color.BLACK);
        g.drawLine(0, originY, w, originY);

        // cart in pixels
        int cartX = originX + (int) (cartPos * SCALE) - CART_WIDTH / 2;
        int cartY = originY - CART_HEIGHT / 2;

        // draw cart
        g.setColor(Color.BLUE);
        g.fillRect(cartX, cartY, CART_WIDTH, CART_HEIGHT);

        // pole
        double angle = -poleAngle; // screen coords: negative angle is clockwise
        int polePivotX = cartX + CART_WIDTH / 2;
        int polePivotY = cartY;
        int x2 = polePivotX + (int) (POLE_HEIGHT * Math.sin(angle));
        int y2 = polePivotY - (int) (POLE_HEIGHT * Math.cos(angle));

        Graphics2D g2 = (Graphics2D) g;
        g2.setColor(Color.RED);
        g2.setStroke(new BasicStroke(POLE_WIDTH));
        g2.drawLine(polePivotX, polePivotY, x2, y2);

        // text
        g.setColor(Color.BLACK);
        g.drawString("Episode (Rendered): " + renderEpisode, 10, 20);
        g.drawString(String.format("Cart Pos = %.2f", cartPos), 10, 35);
        g.drawString(String.format("Cart Vel = %.2f", cartVel), 10, 50);
        g.drawString(String.format("Pole Angle (deg) = %.2f", Math.toDegrees(poleAngle)), 10, 65);
    }

    //--------------------------------
    // Entry point
    //--------------------------------
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            CartPoleEpsilonGreedy cpl = new CartPoleEpsilonGreedy();
            Thread trainThread = new Thread(cpl::train);
            trainThread.start();
        });
    }
}
