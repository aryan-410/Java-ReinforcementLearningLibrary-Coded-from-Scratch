package space;

import java.util.Random;

public class DiscreteActionSpace implements ActionSpace<Integer> {
    private final int size;
    private final Random random;

    public DiscreteActionSpace(int size) {
        this.size = size;
        this.random = new Random();
    }

    public Integer randomAction() {return random.nextInt(size);}
    public int[] getShape() {return new int[]{size, 1};}
}
