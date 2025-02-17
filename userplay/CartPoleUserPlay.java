package userplay;

import environment.CartPole;
import environment.StepResult;
import examples.CartPole.CartPoleVisualizer;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

/**
 * Allows the user to control the CartPole environment interactively.
 * The user must confirm they want to play and then click "Start" to begin.
 * Control the cart with the left and right arrow keys.
 */
public class CartPoleUserPlay extends JFrame implements KeyListener {

    // The CartPole environment instance.
    private CartPole environment;
    // The visualizer that draws the cart and pole.
    private CartPoleVisualizer visualizer;
    // Timer for simulation updates.
    private Timer simulationTimer;
    // This variable holds the current action to apply.
    // We use 0 for left force and 1 for right force.
    private volatile int currentAction = 1;  // default action

    /**
     * Constructor that sets up the window, visualizer, and controls.
     */
    public CartPoleUserPlay() {
        super("CartPole User Play");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create the environment and visualizer.
        environment = new CartPole();
        visualizer = new CartPoleVisualizer();

        // Use BorderLayout for the window.
        setLayout(new BorderLayout());
        add(visualizer, BorderLayout.CENTER);

        // Create a control panel with a Start button.
        JPanel controlPanel = new JPanel();
        JButton startButton = new JButton("Start");
        controlPanel.add(startButton);
        add(controlPanel, BorderLayout.SOUTH);

        // Add this frame as a key listener.
        addKeyListener(this);
        // Ensure the frame is focusable so key events are received.
        setFocusable(true);
        requestFocusInWindow();

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        // When the Start button is clicked, remove the control panel and start the game.
        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controlPanel.setVisible(false);
                startGame();
            }
        });
    }

    /**
     * Starts the game simulation. A Timer is used to update the environment.
     */
    private void startGame() {
        // Reset the environment and update the visualizer with the initial state.
        double[] initialState = environment.reset();
        visualizer.updateState(initialState[0], initialState[2]);

        // Set a default action (right force) if none is provided.
        currentAction = 1;

        // Create a simulation timer (~50 frames per second).
        int delay = 20;
        simulationTimer = new Timer(delay, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Use the currentAction set by the user's key presses.
                StepResult<double[]> result = environment.step(currentAction);
                double[] newState = result.getNextState();
                visualizer.updateState(newState[0], newState[2]);

                // If the environment reaches a terminal state, stop the simulation,
                // show a dialog to inform the user, and then restart the game.
                if (result.isDone()) {
                    simulationTimer.stop();
                    JOptionPane.showMessageDialog(CartPoleUserPlay.this,
                            "Game over! Press OK to restart.",
                            "Game Over",
                            JOptionPane.INFORMATION_MESSAGE);
                    environment.reset();
                    simulationTimer.start();
                }
            }
        });
        simulationTimer.start();
    }

    // --- KeyListener methods ---

    @Override
    public void keyTyped(KeyEvent e) {
        // Not used.
    }

    @Override
    public void keyPressed(KeyEvent e) {
        // Use left arrow key for left force (action 0) and right arrow key for right force (action 1).
        int keyCode = e.getKeyCode();
        if (keyCode == KeyEvent.VK_LEFT) {
            currentAction = 0;
        } else if (keyCode == KeyEvent.VK_RIGHT) {
            currentAction = 1;
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {
        // Optionally, you could set the action to a neutral value here.
        // For this example, we'll simply keep the last action until a new key is pressed.
    }

    /**
     * Main method: asks the user if they want to play, then starts the game.
     */
    public static void main(String[] args) {
        // Show a confirmation dialog.
        int response = JOptionPane.showConfirmDialog(null,
                "Do you want to play the CartPole game?",
                "CartPole User Play",
                JOptionPane.YES_NO_OPTION);
        if (response == JOptionPane.YES_OPTION) {
            // Start the game on the Event Dispatch Thread.
            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    new CartPoleUserPlay();
                }
            });
        } else {
            System.exit(0);
        }
    }
}
