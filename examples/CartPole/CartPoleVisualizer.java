package examples.CartPole;

import javax.swing.*;
import java.awt.*;

/**
 * A JPanel that visually represents the CartPole environment state and shows the episode number.
 */
public class CartPoleVisualizer extends JPanel {
    // Current state values from the environment.
    private double cartPosition; // Environment coordinate for the cart position.
    private double poleAngle;    // Pole angle in radians.

    // Panel dimensions.
    private final int panelWidth = 800;
    private final int panelHeight = 600;

    // Environment x-axis boundaries (should match your environment limits).
    private final double xMin = -2.4;
    private final double xMax = 2.4;

    // Dimensions for the cart (in pixels).
    private final int cartWidth = 60;
    private final int cartHeight = 30;

    // Visual pole length (in pixels).
    private final int poleLength = 150;

    // Episode number to display.
    private int episodeNumber = 0;

    public CartPoleVisualizer() {
        this.setPreferredSize(new Dimension(panelWidth, panelHeight));
    }

    /**
     * Update the visual state from the simulation.
     * @param cartPosition the cart’s x-coordinate (from the environment)
     * @param poleAngle the pole’s angle in radians
     */
    public void updateState(double cartPosition, double poleAngle) {
        this.cartPosition = cartPosition;
        this.poleAngle = poleAngle;
        repaint();
    }

    /**
     * Set the current episode number to be displayed.
     * @param episodeNumber the current episode number.
     */
    public void setEpisodeNumber(int episodeNumber) {
        this.episodeNumber = episodeNumber;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        // Draw background.
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, panelWidth, panelHeight);

        // Draw ground.
        g.setColor(Color.GRAY);
        g.fillRect(0, panelHeight - 50, panelWidth, 50);

        // Map the environment cart position to a pixel coordinate.
        double scale = (panelWidth - cartWidth) / (xMax - xMin);
        int cartX = (int) ((cartPosition - xMin) * scale);
        int cartY = panelHeight - 50 - cartHeight;

        // Draw the cart.
        g.setColor(Color.BLUE);
        g.fillRect(cartX, cartY, cartWidth, cartHeight);

        // The pivot point is at the top center of the cart.
        int pivotX = cartX + cartWidth / 2;
        int pivotY = cartY;

        // Calculate the end point of the pole.
        int poleEndX = (int) (pivotX + poleLength * Math.sin(poleAngle));
        int poleEndY = (int) (pivotY - poleLength * Math.cos(poleAngle));

        // Draw the pole.
        g.setColor(Color.RED);
        g.drawLine(pivotX, pivotY, poleEndX, poleEndY);

        // Draw a small circle at the pivot.
        g.setColor(Color.BLACK);
        g.fillOval(pivotX - 5, pivotY - 5, 10, 10);

        // Draw the episode number in the upper left corner.
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 16));
        g.drawString("Episode: " + episodeNumber, 10, 20);
    }
}
