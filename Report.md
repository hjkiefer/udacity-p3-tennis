# Project 2: Continuous Control

This is the report for the second project in Udacity Deep Learning
specialisation. The report summarizes the solution of the Continuous Control project.

## Results

The environment was solved in 30 episodes, however at this point, the agent was
still improving its policy. At 70 episodes a stable maximum of about 36 points
were achieved.

The averaged agent scores for each episode (averaged over the 20 parallel arms)
is shown in blue. The rolling average for the previous 100 episodes is the red
line, and the dashed green line is the target/solvbed score. 

![Scores](scores.png)

## Learning Algorithm

The learning algorithm used is DDPG (Deep Deterministic Policy Gradient) [1]
with soft-update and an un-prioritized experience replay buffer. This fairly
simple algorithm is an improvement over DQN, when working with continuous
action spaces.


### Hyper-parameters

It took some effort to find a set of working hyper-parameters. However, that
could've been because of a few bugs in my code at that time. The following
parameters seem to work well, and provide robust training, when using the 20
agents environment:

	Buffer_size = 2e5
	batch_size = 64
	gamma = 0.99
	tau = 1e-3
	actor_lr = 1e-4
	critic_lr = 1e-3

On top of that there are 2 parameters which control when a learning step is performed, 

	network_optimize_every_n_step = 16
	learn_iterations = 16

This set when learning is initiated (every 16 steps), and how many times
learning is done (16 times). This could of course have been set to 1 each, to
just run 1 learning iteration every step, but these parameters were changed
multiple times during optimizations.

#### Noise

Finally, after obtaining the action from the critic we add some noise. This
noise has is provided from white (gaussian) noise with a decaying scale. THe
initial noise scale is 0.5 and decays with 0.8 after each episode. It has been
shown in other work, that gaussian noise works just fine when compared to
Ornstein-Uhlenbeck noise [2]

### Neural Netowrks

#### General architecture

The neural networks used are fully connected networks with N hidden layers (build from a list of hidden layer sizes). Each hidden layer is connected to the input layer neurons, e.g.:

	Input(state)  ---> Layer1 --> Layer 2  ----> Layer 3 ...  Layer N---> Output(action)
	    \   \___________________/               /             /
	     \_____________________________________/	         /


This means that layer 1 has e.g. 128 neurons, while layers 2 - 4 has 128 neurons +
the number of input parameters (43). 

The network used resembles residual networks, which have shortcut
connections of similar nature. These shortcuts are made to avoid diminishing
gradients, and aids in training a deeper neural network effeciently.

#### Actor

The actor has 2 hidden layers with 256 and 128 neurons (both hidden layers
connected to input state). All layers (except the output) uses the RELu activation
function. The output neurons are piped through a hypberbolic tangent activation
function.

#### Critic

The critic has 2 inputs, the state and the action. Both are concatenated as a
single input to the first layer. The network has 3 hidden layers all connected
to the inputs. The hidden layers have 256, 256 and 128 neurons which use the
leaky relu activation function. I am not convinced that it was this change that
made the difference (i.e. ensured that the environment was solved). But it was
not thoroughly investigated. The output activation function is a RELu function,
because we can only have positive rewards

## Future Work

This uses the most simple DDPG algorithm without any optimizations or extensions. 

DDPG is an off-policy algorithm, which means that a replay buffer is useful. It
also means that cause-and-effect is not as easily learned through multiple
steps. Here we use a 1-step bootstrap (Temporal-difference loss). For an
on-policy method, like A3C or A2C it would've been possible to record longer
trajectories, and more accurately record the expected return. 

In this solution I used the 20 agents environment as a way to sample 20 actions
at the same time from the same actor, this would also have been possible with
A2C, so I would probably have started with that algorithm for an on-policy
agent.

# References

[1] Lillicrap, T. P. et al. Continuous control with deep reinforcement learning. arXiv:1509.02971v6

[2] Fujimoto, S. et al. Addressing Function Approximation Error in Actor-Critic Methods. arXiv:1802.09477v3
