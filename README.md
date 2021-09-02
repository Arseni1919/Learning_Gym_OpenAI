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

### Spaces

So, you can see the similarity between actions and observations, and how they have found their representation in Gym's classes.
Let's look at a class diagram:

![spaces](static/spaces.png)

The basic abstract class Space includes two methods that are relevant to us:

- `sample()`: This returns a random sample from the space

- `contains(x)`: This checks whether the argument, x, belongs to the space's
domain

Both of these methods are abstract and reimplemented in each of the Space
subclasses:

- The `Discrete` class represents a mutually exclusive set of items, numbered
from 0 to n – 1. Its only field, `n`, is a count of the items it describes. For
example, `Discrete(n=4)` can be used for an action space of four directions
to move in `[left, right, up, or down]`.

- The `Box` class represents an n-dimensional tensor of rational numbers
with intervals `[low, high]`. For instance, this could be an accelerator pedal
with one single value between 0.0 and 1.0, which could be encoded by
`Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)`(the `shape`
argument is assigned a tuple of length 1 with a single value of 1, which
gives us a one-dimensional tensor with a single value). The `dtype` parameter
specifies the space's value type and here we specify it as a NumPy 32-bit
float. Another example of `Box` could be an Atari screen observation (we
will cover lots of Atari environments later), which is an RGB (red, green,
and blue) image of size 210×160: `Box(low=0, high=255, shape=(210, 160,
3), dtype=np.uint8)`. In this case, the `shape` argument is a tuple of three
elements: the first dimension is the height of the image, the second is the
width, and the third equals 3, which all correspond to three color planes
for red, green, and blue, respectively. So, in total, every observation is
a three-dimensional tensor with 100,800 bytes.

- The final child of `Space` is a `Tuple` class, which allows us to combine
several Space class instances together. This enables us to create action
and observation spaces of any complexity that we want. For example,
imagine we want to create an action space specification for a car. The car
has several controls that can be changed at every timestamp, including the
steering wheel angle, brake pedal position, and accelerator pedal position.
These three controls can be specified by three float values in one single `Box`
instance. Besides these essential controls, the car has extra discrete controls,
like a turn signal (which could be off, right, or left) or horn (on or off). To
combine all of this into one action space specification class, we can create
`Tuple(spaces=(Box(low=-1.0, high=1.0, shape=(3,), dtype=np.
float32), Discrete(n=3),Discrete(n=2)))`. This flexibility is rarely used;
for example, in this book, you will see only the Box and Discrete actions
and observation spaces, but the Tuple class can be useful in some cases.

---

There are other Space subclasses defined in Gym, but the preceding three are the
most useful ones. All subclasses implement the `sample()` and `contains()` methods.
The `sample()` function performs a random sample corresponding to the `Space` class
and parameters. This is mostly useful for action spaces, when we need to choose the
random action. The `contains()` method verifies that the given arguments comply
with the `Space` parameters, and it is used in the internals of Gym to check an agent's
actions for sanity. For example, `Discrete.sample()` returns a random element from
a discrete range, and `Box.sample()` will be a random tensor with proper dimensions
and values lying inside the given range.

Every environment has two members of type `Space`: the action_space and
observation_space. This allows us to create generic code that could work with
any environment. Of course, dealing with the pixels of the screen is different from
handling discrete observations (as in the former case, we may want to preprocess
images with convolutional layers or with other methods from the computer vision
toolbox); so, most of the time, this means optimizing the code for a particular
environment or group of environments, but Gym doesn't prevent us from writing
generic code.












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









