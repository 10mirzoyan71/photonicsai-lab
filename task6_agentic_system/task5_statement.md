Task 5. PINN (Physics Informed Neural Network) for the Silverbox oscillator

The Silverbox is a laboratory electronic circuit built to act as a Duffing oscillator: a second-order system with a spring that stiffens as the displacement grows. The public record is the Silverbox benchmark of Wigren and Schoukens (European Control Conference, 2013), file SNLS80mV.mat (it is in your working directory). Channel V1 is the input voltage u(t). Channel V2 is the output voltage y(t). Both are sampled at fs = 610.35 Hz, dt = 1/fs. The file contains 131,072 samples. The estimation experiment is a random-phase multisine, and it begins at sample 40650 (indices into the file, starting at 0).

The excerpt. Use the 4096 samples starting at index 42650, that is the slice [42650 : 46746]. Subtract the mean of u on this excerpt and the mean of y on this excerpt, and use those centered signals afterwards. The first 3072 samples are the training record, about 5.03 s. The last 1024 samples are the test record, about 1.68 s. On the test record the input u is known; the output y is what the model has to produce.

The differential equation. With the centered output y and centered input u, the circuit is modeled by
    y'' + a y' + b y + c y^3 = g u(t).                                          (1)
a is the damping (a few tens of 1/s on this circuit); b is the linear stiffness in 1/s^2 (the output of this excerpt is dominated by frequencies near 70 Hz); c is the cubic stiffness (at the peaks of this excerpt, |y| near 0.2 V, the cubic force is a visible fraction of the linear force); g is an input gain. They are not the weights of the network. The acceleration is produced by a network that has to represent this force.

The network. The acceleration is a multilayer fully connected network, implemented in PyTorch. You choose the depth, the widths, the activation, and how the inputs and the output are scaled. With velocity v = y', the state is (y, v) and
    y' = v,   v' = alpha_theta(y, v, u).                                         (2)
Integrate that state at step dt, driven by the measured input. Classical Runge-Kutta of order 4 is a sound choice. The excerpt lasts 6.7 s near 70 Hz, several hundred cycles, so the integrator is what draws the waveform.

The metric. After the weights are fixed, do one free run of the whole excerpt. Start at the first sample, with the measured position and a finite-difference velocity, and integrate through the training samples and on into the test samples. Keep the same state when the test segment begins. On the test segment the rollout may read u, and it may not read y. The performance is the root-mean-square error of this free run on the 1024 test samples, RMSE = sqrt(mean((y_hat - y)^2)). Predicting zero scores about 0.06 V, the baseline. Also compute the same free-run RMSE on the 3072 training samples.

Deliver: the free-run TEST_RMSE and TRAIN_RMSE printed by a script that actually ran, and a plot of measured against predicted output on the test segment.
