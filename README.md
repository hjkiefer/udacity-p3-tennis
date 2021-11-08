# Project 2: Continuous Control

## Introduction

This project is about training an agent to control a robot arm to follow a
moving target.  The environment consists of a robot arm, which can be rotated
along the base a two joints. The tip of the arm has to be within a target
sphere, which moves as time progresses

The agent is awarded +0.1 points for every time step where the tip of the arm
is within the target volume.

The agent have 4 continuous actions which controls the amount of force applied
at different points.

Each episode consist of 1000 timesteps and the agent must get an
average score of +30 over 100 consecutive episodes in order to solve the
environment.

## Setting up your environment

It has not been possible to setup the environment locally due to setup issues
with old versions of the unity environment, so for me, it was necessary to run it in the
Udacity provided cloud VM which has all dependencies already installed. 

Optionally, you can attempt to install the dependencies yourself by following the guide here
[udacity drl github repo](https://github.com/udacity/deep-reinforcement-learning#dependencies)

## Training the agent

Run the code in the [Continuous_Control.ipynb](Continuous_Control.ipynb) to train the agent.
This loads 

* torch networks from [networks.py](networks.py) 
* agent from [agents.py](agents.py)
* replay buffer from [replay_buffer.py](replay_buffer.py) 
* a status printer from [status_printer.py](status_printer.py), which updates the status after each iteration 

And finally runs a training loop which ends when the environment has been solved.
