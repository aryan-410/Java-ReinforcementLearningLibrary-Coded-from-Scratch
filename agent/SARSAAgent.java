package agent;

public class SARSAAgent extends Agent<double[], Integer> {

    boolean yeyeye;

    public SARSAAgent () {
        super(0.5, 6);
        this.yeyeye = true;
    }

    @Override
    public Integer chooseAction(double[] state) {
        return 1;
    }

    @Override
    public void learn(double[] state, Integer action, double reward, double[] nextState, boolean done){

    }
}
