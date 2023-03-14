# using chainerRL cuz it seems easy!
# I wanna use NEAT but it's kinda difficult for me...

# chainerRL
# https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb

# \Python\Python311\Lib\site-packages\chainerrl\agents\ddpg.py のminibatch_size=32をminibatch_size=3に変更

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from time import sleep

import learning_gym

env = learning_gym.KSPgym()
print('observation space:', env.observation_space)
print('action space:', env.action_space)

"""
obs = env.reset()
env.render()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)
"""
""" DQNのときのやつ
class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        # Args:
        #     x (ndarray or chainer.Variable): An observation
        #     test (bool): a flag indicating whether it is in test mode
        
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)


# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
"""
### SAC
def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = F.split_axis(x, 2, axis=1)
        log_scale = F.clip(log_scale, -20., 2.)
        var = F.exp(log_scale * 2)
        return chainerrl.distribution.SquashedGaussianDistribution(
            mean, var=var)

action_space = env.action_space
action_size = action_space.n   
winit = chainer.initializers.GlorotUniform()
winit_policy_output = chainer.initializers.GlorotUniform(1)
policy = chainer.Sequential(
    L.Linear(None, 256, initialW=winit),
    F.relu,
    L.Linear(None, 256, initialW=winit),
    F.relu,
    L.Linear(None, action_size * 2, initialW=winit_policy_output),
    squashed_diagonal_gaussian_head,
)
policy_optimizer = chainer.optimizers.Adam(3e-4).setup(policy)

def concat_obs_and_action(obs, action):
    """Concat observation and action to feed the critic."""
    return F.concat((obs, action), axis=-1)

def make_q_func_with_optimizer():
    q_func = chainer.Sequential(
        concat_obs_and_action,
        L.Linear(None, 256, initialW=winit),
        F.relu,
        L.Linear(None, 256, initialW=winit),
        F.relu,
        L.Linear(None, 1, initialW=winit),
    )
    q_func_optimizer = chainer.optimizers.Adam(3e-4).setup(q_func)
    return q_func, q_func_optimizer

q_func1, q_func1_optimizer = make_q_func_with_optimizer()
q_func2, q_func2_optimizer = make_q_func_with_optimizer()

###

# Set the discount factor that discounts future rewards.
gamma = 0.99

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.2, random_action_func=env.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
"""DQNのときのやつ
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=32, update_interval=1,
    target_update_interval=3, phi=phi)
"""
def burnin_action_func():
    """Select random actions until model is updated one or more times."""
    return action_space.sample().astype(np.float32)
agent = chainerrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        replay_buffer,
        gamma=0.99,
        replay_start_size= 8, # args.replay_start_size,
        # gpu=args.gpu,
        minibatch_size= 1, # args.batch_size,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer=chainer.optimizers.Adam(3e-4),
    )

#####


print("input 'train' for training")
model_name = "2"

# agent.load(f'models/{model_name}/agent_final')

if input() == "train":

    print("starting training in 3 seconds...")
    sleep(3)

    n_episodes = 500
    # max_episode_len = 200
    for i in range(1, n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step
        while not done:
            action = agent.act_and_train(obs, reward)
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
        if i % 1 == 0:
            print('episode:', i,
                'R:', R,
                'statistics:', agent.get_statistics())
            agent.save(f'models/{model_name}/agent_{i}')
        agent.stop_episode_and_train(obs, reward, done)
    print('Finished.')

    # Save an agent to the 'agent' directory
    agent.save(f'models/{model_name}/agent_final')

# Uncomment to load an agent from the 'agent' directory
#agent.load('models/1/agent_final')

obs = env.reset()
done = False
R = 0
t = 0
while not done:
    env.render()
    action = agent.act(obs)
    obs, r, done, _ = env.step(action)
    R += r
    t += 1
print('R:', R)
agent.stop_episode()