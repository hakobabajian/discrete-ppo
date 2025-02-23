# Discrete Proximal Policy Optimization 
## Objective
This project aims to train a discrete ppo, a reinforement learning model, to control the pitch authority of an aircraft in the popular aerospace simulator, Kerbal Space Program. The goal is to train the model such that it can pilot the aircraft to take-off from the runway, and maintain a reasonable altitude without going out of bounds for 30 seconds.

#### Dependencies
This project uses the following python libraries:
- pytorch: for developing and training neural networks
- krpc: for programmatic control of aircraft in the aerospace simulation software

## Discrete vs. Continuous Action Spaces
This Proximal Policy Optimization (PPO) model is defined as "discrete" (as opposed to "continuous"), because the action (output) of its actor network is an integer value instead of a float value. The discrete action space used in this project is defined below:

Discrete Action Space: [-1, 0, 1] for "pitch more down", "do nothing", and "pitch more up"

This is in contrast to a continuous action space which can vary the change to the pitch authority more sensitively: [float] (0,1).

## Results
After experimenting with training the discrete PPO in this project and training a continuous PPO in an adjacent project, it became clear that the continuous PPO was far better at learning to control the pitch authority of the aircraft. Link to the continuous PPO repo:

[continuous-ppo](https://github.com/hakobabajian/continuous-ppo.git)

The ReadMe from that repository includes much more thorough documnetation of technical details, methods, and results.

## Conclusion
After appreciating the short-comings of discrete action spaces when training this PPO, progress on this project stopped. The adjacent project 'continuous-ppo', linked above, took full advantange of a continuous action space which proved to be far more effective for training a deep learning model in the context of aircraft control.
