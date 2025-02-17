package agent;

import A2C.TRPOActor;
import A2C.TRPOCritic;
import environment.CartPole;
import environment.StepResult;
import util.Transition;

import java.util.ArrayList;
import java.util.List;

public class TRPOAgent {
    private CartPole env;
    private TRPOActor actor;
    private TRPOCritic critic;
    private double gamma;
    private int stateDim;
    private int actionDim;

    // TRPO hyperparameters:
    private double maxKL;      // maximum allowed KL divergence (δ)
    private int cgIterations;  // number of conjugate gradient iterations
    private double cgTolerance;

    public TRPOAgent(CartPole env, double gamma, double maxKL, int cgIterations, double cgTolerance, double criticLr) {
        this.env = env;
        this.gamma = gamma;
        this.maxKL = maxKL;
        this.cgIterations = cgIterations;
        this.cgTolerance = cgTolerance;
        this.stateDim = 4;   // for CartPole
        this.actionDim = 2;
        actor = new TRPOActor(stateDim, actionDim);
        critic = new TRPOCritic(stateDim, criticLr);
    }

    // Conjugate gradient solver to solve A*x = b.
    public double[] conjugateGradient(java.util.function.Function<double[], double[]> A, double[] b, int iterations, double tol) {
        int n = b.length;
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = 0.0;
        }
        double[] r = b.clone();
        double[] p = r.clone();
        double rsold = dot(r, r);
        for (int i = 0; i < iterations; i++) {
            double[] Ap = A.apply(p);
            double alpha = rsold / (dot(p, Ap) + 1e-8);
            for (int j = 0; j < n; j++) {
                x[j] += alpha * p[j];
                r[j] -= alpha * Ap[j];
            }
            double rsnew = dot(r, r);
            if (rsnew < tol) break;
            for (int j = 0; j < n; j++) {
                p[j] = r[j] + (rsnew / rsold) * p[j];
            }
            rsold = rsnew;
        }
        return x;
    }

    private double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // Train the agent.
    public void train(int episodes) {
        List<Transition> trajectory = new ArrayList<>();
        for (int episode = 0; episode < episodes; episode++) {
            trajectory.clear();
            double[] state = env.reset();
            double totalReward = 0.0;
            while (!env.isDone()) {
                Transition t = new Transition();
                t.state = state.clone();
                t.action = actor.selectAction(state, t);
                t.value = critic.value(state);
                StepResult<double[]> result = env.step(t.action);
                t.reward = result.getReward();
                totalReward += t.reward;
                state = result.getState().clone();
                trajectory.add(t);
            }

            // Compute discounted returns and advantages.
            double G = 0.0;
            for (int i = trajectory.size() - 1; i >= 0; i--) {
                Transition t = trajectory.get(i);
                G = t.reward + gamma * G;
                t.returnG = G;
                t.advantage = t.returnG - t.value;
            }

            // Update the critic using the batch.
            critic.updateBatch(trajectory);

            // Compute the policy gradient "g".
            double[] g = actor.computePolicyGradient(trajectory);

            // Define the function A(v) = F*v (Fisher vector product).
            java.util.function.Function<double[], double[]> A = (double[] v) -> actor.fisherVectorProduct(v, trajectory);

            // Solve for the step direction d using conjugate gradient.
            double[] d = conjugateGradient(A, g, cgIterations, cgTolerance);

            // Compute d^T F d.
            double[] Fd = actor.fisherVectorProduct(d, trajectory);
            double dFd = dot(d, Fd);
            double stepSize = Math.sqrt(2 * maxKL / (dFd + 1e-8));

            // Candidate update: deltaTheta = stepSize * d.
            double[] deltaTheta = new double[d.length];
            for (int i = 0; i < d.length; i++) {
                deltaTheta[i] = stepSize * d[i];
            }

            // Save old parameters.
            double[] oldTheta = actor.flattenParameters();
            // Backtracking line search.
            double stepFraction = 1.0;
            boolean found = false;
            double oldSurrogate = computeSurrogateLoss(trajectory, oldTheta);
            double[] newTheta = null;
            for (int i = 0; i < 10; i++) {
                newTheta = new double[oldTheta.length];
                for (int j = 0; j < oldTheta.length; j++) {
                    newTheta[j] = oldTheta[j] + stepFraction * deltaTheta[j];
                }
                actor.setParameters(newTheta);
                double kl = actor.computeAverageKL(trajectory);
                double newSurrogate = computeSurrogateLoss(trajectory, newTheta);
                if (kl <= maxKL && newSurrogate > oldSurrogate) {
                    found = true;
                    break;
                }
                stepFraction *= 0.5;
            }
            if (!found) {
                // If no acceptable step is found, revert.
                actor.setParameters(oldTheta);
            }

            System.out.println("Episode " + episode + ": Total Reward = " + totalReward);
        }
    }

    // Compute the surrogate loss: the average of exp(newLogProb - oldLogProb)*advantage.
    private double computeSurrogateLoss(List<Transition> batch, double[] theta) {
        // Save current parameters.
        double[] current = actor.flattenParameters();
        actor.setParameters(theta);
        double surrogate = 0.0;
        for (Transition t : batch) {
            double[] probs = actor.forward(t.state);
            double newLogProb = Math.log(probs[t.action] + 1e-8);
            surrogate += Math.exp(newLogProb - t.oldLogProb) * t.advantage;
        }
        surrogate /= batch.size();
        // Restore original parameters.
        actor.setParameters(current);
        return surrogate;
    }
}
