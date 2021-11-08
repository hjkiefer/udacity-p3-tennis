# Project 2: Continuous Control

This is the report for the second project in Udacity Deep Learning
specialisation. The report summarizes the solution of the Continuous Control project.

## Results

The environment was solved in 30 episodes.
The agent scores along with rolling average is seen in the figure below:

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

During the exploration track I found on several occasions, that the agent had a
hard time picking up speed on training. In particular, it seemed that for a
long time, episodes with 0 reward kept appearing. It would be interesting to
check whether prioritized experience replay could avoid such episodes, or even
improve learning speed. 

It would also be a blast to try different methods for getting the overall
highest performance, not just beating a score of 13, but trying to maximise the
overall score, say on a 2000 episode training run. Here it would be possible to
benchmark different algorithm improvements against each other, fx Double DQN [2] or
dueling DQN [3], or mixtures of these with/without prioritized experience replay.

# References

[1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236

[2] van Hasselt, H. et al. Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461

[3] Wang, Z. et al. Dueling Network Architectures for Deep Reinforcement Learning. arXiv:1511.06581
