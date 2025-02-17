package A2C;

import java.util.List;
import java.util.Random;

public class TRPOActor {
    private int stateDim;
    private int actionDim;
    public double[][] weights; // policy parameters (shape: [actionDim x stateDim])
    public double[] biases;    // biases for each action
    private Random random;

    public TRPOActor(int stateDim, int actionDim) {
        this.stateDim = stateDim;
        this.actionDim = actionDim;
        this.random = new Random();
        weights = new double[actionDim][stateDim];
        biases = new double[actionDim];
        // Initialize weights and biases to small random values.
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim; j++) {
                weights[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
            biases[i] = 0.0;
        }
    }

    // Forward pass: compute logits = W * state + b, then softmax.
    public double[] forward(double[] state) {
        double[] logits = new double[actionDim];
        for (int i = 0; i < actionDim; i++) {
            double sum = biases[i];
            for (int j = 0; j < stateDim; j++) {
                sum += weights[i][j] * state[j];
            }
            logits[i] = sum;
        }
        return softmax(logits);
    }

    private double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) {
            if (l > max) max = l;
        }
        double sum = 0.0;
        double[] exp = new double[actionDim];
        for (int i = 0; i < actionDim; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }
        double[] probs = new double[actionDim];
        for (int i = 0; i < actionDim; i++) {
            probs[i] = exp[i] / sum;
        }
        return probs;
    }

    // Select an action and store the old distribution in the transition.
    public int selectAction(double[] state, Transition transition) {
        double[] probs = forward(state);
        transition.oldProbs = probs.clone();
        double r = random.nextDouble();
        double cumulative = 0.0;
        int action = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                action = i;
                break;
            }
        }
        transition.oldLogProb = Math.log(probs[action] + 1e-8);
        return action;
    }

    // Flatten the actor parameters (first weights, then biases) into a 1D vector.
    public double[] flattenParameters() {
        int total = actionDim * stateDim + actionDim;
        double[] theta = new double[total];
        int index = 0;
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim; j++) {
                theta[index++] = weights[i][j];
            }
        }
        for (int i = 0; i < actionDim; i++) {
            theta[index++] = biases[i];
        }
        return theta;
    }

    // Set parameters from a flattened parameter vector.
    public void setParameters(double[] theta) {
        int index = 0;
        for (int i = 0; i < actionDim; i++) {
            for (int j = 0; j < stateDim; j++) {
                weights[i][j] = theta[index++];
            }
        }
        for (int i = 0; i < actionDim; i++) {
            biases[i] = theta[index++];
        }
    }

    // Returns the dimensionality of the parameter vector.
    public int getParameterDimension() {
        return actionDim * stateDim + actionDim;
    }

    // Compute the (averaged) policy gradient over a batch.
    // For a softmax policy, the gradient of log π(a|s) is (one_hot(a) - probs)*state.
    // We then weight by the advantage.
    public double[] computePolicyGradient(List<Transition> batch) {
        int dim = getParameterDimension();
        double[] grad = new double[dim];
        int batchSize = batch.size();
        for (Transition t : batch) {
            double[] probs = forward(t.state);
            int index = 0;
            for (int i = 0; i < actionDim; i++) {
                for (int j = 0; j < stateDim; j++) {
                    double indicator = (i == t.action) ? 1.0 : 0.0;
                    double g = (indicator - probs[i]) * t.state[j] * t.advantage;
                    grad[index++] += g;
                }
            }
            // Gradients for biases.
            for (int i = 0; i < actionDim; i++) {
                double indicator = (i == t.action) ? 1.0 : 0.0;
                double g = (indicator - probs[i]) * t.advantage;
                grad[index++] += g;
            }
        }
        // Average over the batch.
        for (int i = 0; i < grad.length; i++) {
            grad[i] /= batchSize;
        }
        return grad;
    }

    // Compute the gradient of the average KL divergence over the batch.
    // For each transition, the KL divergence between the old and new policy is:
    // KL(old || new) = sum_i oldProbs[i] * (log(oldProbs[i]) - log(newProbs[i])).
    // We approximate its gradient with respect to the parameters.
    public double[] computeKLGradient(List<Transition> batch) {
        int dim = getParameterDimension();
        double[] grad = new double[dim];
        int batchSize = batch.size();
        for (Transition t : batch) {
            double[] newProbs = forward(t.state);
            int index = 0;
            for (int i = 0; i < actionDim; i++) {
                // For the weights of action i.
                for (int j = 0; j < stateDim; j++) {
                    // A simplified approximation: error = newProbs[i] - oldProbs[i].
                    double error = newProbs[i] - t.oldProbs[i];
                    grad[index++] += error * t.state[j];
                }
            }
            // For biases.
            for (int i = 0; i < actionDim; i++) {
                double error = newProbs[i] - t.oldProbs[i];
                grad[index++] += error;
            }
        }
        // Average over batch.
        for (int i = 0; i < grad.length; i++) {
            grad[i] /= batchSize;
        }
        return grad;
    }

    // Compute the average KL divergence over the batch.
    public double computeAverageKL(List<Transition> batch) {
        double klSum = 0.0;
        for (Transition t : batch) {
            double[] newProbs = forward(t.state);
            for (int i = 0; i < actionDim; i++) {
                klSum += t.oldProbs[i] * (Math.log(t.oldProbs[i] + 1e-8) - Math.log(newProbs[i] + 1e-8));
            }
        }
        return klSum / batch.size();
    }

    // Approximate the Fisher–vector product: F*v ≈ (∇²KL) * v.
    // We use a finite-difference approximation on the KL gradient.
    public double[] fisherVectorProduct(double[] v, List<Transition> batch) {
        double r = 1e-5;
        double[] theta = flattenParameters();
        // Compute theta_plus = theta + r*v.
        double[] thetaPlus = new double[theta.length];
        for (int i = 0; i < theta.length; i++) {
            thetaPlus[i] = theta[i] + r * v[i];
        }
        // Set parameters to theta_plus and compute the KL gradient.
        setParameters(thetaPlus);
        double[] gradPlus = computeKLGradient(batch);
        // Reset to original theta.
        setParameters(theta);
        double[] grad = computeKLGradient(batch);
        double[] fvp = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            fvp[i] = (gradPlus[i] - grad[i]) / r;
        }
        // Add a small damping term for numerical stability.
        double damping = 1e-3;
        for (int i = 0; i < fvp.length; i++) {
            fvp[i] += damping * v[i];
        }
        return fvp;
    }
}
