# Learning Gym Environments (OpenAI)

```python
#!/usr/bin/env python
# prints all the names of envs that we have
from gym import envs
envids = [spec.id for spec in envs.registry.all()]
counter = 0
for envid in sorted(envids):
    counter += 1
#     print(envid)
print(counter)
```

## The OpenAI Gym API

The Python library called Gym was developed and has been maintained by OpenAI ([www.openai.com](https://www.openai.com/)).
The main goal of Gym is to provide a rich collection of environments for RL experiments using a unified interface.
So, it is not surprising that the central class in the library is an environment, which is called Env.
Instances of this class expose several methods and fields that provide the required information about its capabilities.
At a high level, every environment provides these pieces of information and functionality:

- A set of actions that is allowed to be executed in the environment. Gym supports both discrete and continuous actions, as well as their combination

- The shape and boundaries of the observations that the environment provides the agent with

- A method called step to execute an action, which returns the current observation, the reward, and the indication that the episode is over

- A method called reset, which returns the environment to its initial state and obtains the first observation

Let's now talk about these components of the environment in detail.

### The action space

As mentioned, the actions that an agent can execute can be discrete, continuous, or
a combination of the two. Discrete actions are a fixed set of things that an agent can
do, for example, directions in a grid like left, right, up, or down. Another example
is a push button, which could be either pressed or released. Both states are mutually
exclusive, because a main characteristic of a discrete action space is that only one
action from a finite set of actions is possible.
A continuous action has a value attached to it, for example, a steering wheel, which
can be turned at a specific angle, or an accelerator pedal, which can be pressed with
different levels of force. A description of a continuous action includes the boundaries
of the value that the action could have. In the case of a steering wheel, it could be
from −720 degrees to 720 degrees. For an accelerator pedal, it's usually from 0 to 1.
Of course, we are not limited to a single action; the environment could take multiple
actions, such as pushing multiple buttons simultaneously or steering the wheel and
pressing two pedals (the brake and the accelerator). To support such cases, Gym
defines a special container class that allows the nesting of several action spaces into
one unified action.

### The observation space
Observations are pieces of information that an environment provides the agent with, on every timestamp, besides the reward.
Observations can be as simple as a bunch of numbers or as complex as several multidimensional tensors containing color images from several cameras.
An observation can even be discrete, much like action spaces.
An example of a discrete observation space is a lightbulb, which could be in two states – on or off, given to us as a Boolean value.

Spaces
So, you can see the similarity between actions and observations, and how they have found their representation in Gym's classes.
Let's look at a class diagram:

![spaces](static/spaces.png)












## Credits:

- [Gym OpenAI wiki | GitHub](https://github.com/openai/gym/wiki)
- [Environments | OpenAI](http://gym.openai.com/envs/#classic_control)
- [Docs | OpenAI](http://gym.openai.com/docs/)
- [Creating new Gym Env | OpenAI](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
- [Create custom gym environments from scratch — A stock market example | Adam King](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()
- []()









