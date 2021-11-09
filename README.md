# Project 3: Tennis 

## Introduction

This project is about training two collaborative agents to play tennis.  The
environment consists of two rackets that can be controlled in 2 directions.  In
addition the environment contains a ball. An episode ends when the ball hits
the ground or is shot off bounds

The agent is awarded +0.1 points for every time the ball is passed to the other
side. A negative -0.05 is awarded if the ball is dropped, or shot out of
bounds.

The agent have 2 continuous actions which controls the movement in the 2
directions, up-down/left-right.

The state space consists of the racket position and speed, along with the speed
and position of the ball an oposing racket

## Setting up your environment

It has not been possible to setup the environment locally due to setup issues
with old versions of the unity environment, so for me, it was necessary to run
it in the Udacity provided cloud VM which has all dependencies already
installed. 

Optionally, you can attempt to install the dependencies yourself by following
the guide here
[udacity drl github repo](https://github.com/udacity/deep-reinforcement-learning#dependencies)

## Training the agent

Run the code in the [Tennis.ipynb](Tennis.ipynb) to train the agent.
This loads 

* torch networks from [networks.py](networks.py) 
* agent from [agents.py](agents.py)
* replay buffer from [replay_buffer.py](replay_buffer.py) 
* a status printer from [status_printer.py](status_printer.py), which updates the status after each iteration 

And finally runs a training loop which ends when the environment has been solved.
