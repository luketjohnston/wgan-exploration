# The problem

In RL environments like Montezuma's Revenge, rewards are rare enough that random exploration doesn't
find many of them. When this is the case, RL algorithms that rely entirely on random exploration
to find initial (and frontier) rewards don't perform well. 

One strategy to solve this is **intrinsic rewards** - using some additional "fake" rewards
beyond those directly supplied by the environment to train the RL agent. For example, if the
agent receives intrinsic rewards for reaching states that it has not encountered often, then
it can learn to explore more efficiently than a random search. TODO cite?

One difficulty of the above approach (rewarding seldom seen states) is: how do we know how many
times the agent has visited a state? In environments with large state spaces, it is infeasible
to maintains counts for every single state, and even if we could, most states would have counts
of 1 or zero. There are many ways to solve these problems - for example, in [1], a simple 
pixel density model is used to estimate *pseudo-counts* for states. 

# The project 

In this project, my initial idea was to use autoencoder reconstruction loss as an intrinsic reward,
with the idea being that if the autoencoder can't reconstruct a state, it probably means the
autoencoder hasn't been trained on similar states very often, so it would work as a proxy for
state visitation count. However, the excellent paper "Exploration by Random Network Distillation" [1]
has done that already so I decided to try
a different but similar approach: train a WGAN on states from the environment, and then
use an intrinsic reward of *the absolute value of the critic score*. The idea is that 
states that the agent has seen many times, the WGAN will have been trained on a lot, and be good
at generating similar states. So, the critic score should be close to 0. States that the agent
hasn't seen many times will not have this quality, so the critic score could be larger or smaller.
Another motivation of using the WGAN instead of an autoencoder is that WGANs are much easier to train
- in my initial tinkerings, the autoencoder reconstructions never looked that great, while the WGAN
converged pretty quickly to reasonable generated images.

# Implementation
I implemented PPO for the agent. See its dedicated repo for more details:

https://github.com/luketjohnston/ppo 

I implemented WGAN-GP to use the critic score for the intrinsic rewards. See my WGAN-GP repo
for more details:

https://github.com/luketjohnston/wgan-gp

The only other really important implementation details involves how I combine the intrinsic
and extrinsic rewards. Initially I just added them together (normalizing intrinsic reward 
by their rolling statistics, and multiplying them by 0.01 so they would be significantly less
important than extrinsic rewards (which are all binned to {+1,0,-1})) 
when computing the generalized advantage estimate. However this greatly disincentivize the agent
from terminating episodes, since now it constantly gets positive rewards from just staying alive. 
After inspecting policy learned with this approach, it was clear that the agent was forgoing opportunities
to explore (that might result in its demise) for a safe strategy of just sticking around an area with 
locally high intrinsic rewards. That "safe" area would change as training progressed,
but it never moved past the "dangerous" areas of the environment (like the rolling skull at the bottom of 
the first room). 

In [1], intrinsic and extrinsic rewards are modeled with two distinct value heads, and intrinsic and 
extrinsic GAEs are computed separately. This allows the rollouts of intrinsic and extrinsic
rewards to be treated differently - specifically, they do not terminate intrinsic reward rollout
on episode termination. Intrinsic rewards "roll over" from one episode to another, so they don't disincentivize
the agent from exploring potentially dangerous areas. So, I implemented this approach following that paper. 

# Results
Unfortunately, my theory that the absolute value of the critic score would be high in unfamiliar areas was completely 
wrong. Upon inspection (after one training run of 7 million frames), it was quite high (7) in areas near the start of 
the game, around 7 for all of the first room, and it dropped to around 1 or -1 for unexplored rooms. The lesson I learned 
from this is that I should have just
manually inspected critic scores on unfamiliar states when training the WGAN from random exploration (before spending
a lot of time making sure PPO was working - although that was certainly rewarding as its own project). In general this 
is a good rule for testing new ideas - try to falsify them in the simplest ways first, before doing a lot of work
under some assumptions that turn out to be incorrect.

However, the above observation seems to suggest that an intrinsic reward of negative the critic score could work 
(or possibly the MSE of the critic score from its mean). I tried the first approach and it didn't work either, the
policy I got after 16M frames was just trying to terminate the episode as quickly as possible. This seems
to suggest that something weird is happening on episode termination... I don't know why these intrinsic rewards
would lead to that policy. Perhaps the critic score is highest on the initial state, so the agent is just
trying to get back to the initial state as quickly as possible every time? In any case, I've kinda lost interest
in the whole WGAN-for-intrinsic-rewards idea... the Exploration by Random Network Distillation is a much
more elegant solution to this problem in my opinion. I may try to implement that next, since I've done
a lot of the work in this project already.

TODO add some videos of the different learned policies
TODO potentially try the last idea for intrinsic reward (MSE of critic score from its own rolling average).
TODO add some states generated by the WGAN


[Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/pdf/1606.01868.pdf)

[Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
