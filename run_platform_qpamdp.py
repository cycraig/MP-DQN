import logging
import click
import time
import numpy as np
import os
import gym
import gym_platform

from agents.qpamdp import QPAMDPAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from common.wrappers import QPAMDPScaledParameterisedActionWrapper
from gym.wrappers import Monitor
from common.wrappers import ScaledStateWrapper


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            action = agent.act(state)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)

    return np.array(returns)


@click.command()
@click.option('--seed', default=7, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=100, help='Episodes over which to evaluate after training.', type=int)
@click.option('--parameter-rollouts', default=50, help='Number of rollouts per parameter update.', type=int)  # default 50, 25 best
@click.option('--scale', default=False, help='Scale inputs and actions.', type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--title', default="QPAMDP", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, parameter_rollouts, scale, initialise_params, save_dir, title):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    alpha_param = 1.0
    variances = [0.1, 0.1, 0.01]
    initial_params = [3., 10., 400.]
    env = gym.make('Platform-v0')
    dir = os.path.join(save_dir, title)
    if scale:
        env = ScaledStateWrapper(env)
        variances = [0.0001, 0.0001, 0.0001]
        for a in range(env.action_space.spaces[0].n):
            initial_params[a] = 2. * (initial_params[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
        env = QPAMDPScaledParameterisedActionWrapper(env)
        alpha_param = 0.1

    env = Monitor(env, directory=os.path.join(dir,str(seed)), video_callable=False, write_upon_reset=False, force=True)

    env.seed(seed)
    np.random.seed(seed)

    act_obs_index = [0, 1, 2, 3]
    param_obs_index = None
    discrete_agent = SarsaLambdaAgent(env.observation_space.spaces[0], env.action_space.spaces[0], alpha=1.0,
                                      gamma=0.999, temperature=1.0, cooling=0.995, lmbda=0.5, order=6,
                                      scale_alpha=True, use_softmax=True, seed=seed,
                                      observation_index=act_obs_index, gamma_step_adjust=True)
    agent = QPAMDPAgent(env.observation_space.spaces[0], env.action_space, alpha=alpha_param,
                        initial_action_learning_episodes=10000, seed=seed, action_obs_index=act_obs_index,
                        parameter_obs_index=param_obs_index, action_relearn_episodes=1000, variances=variances,
                        parameter_updates=180, parameter_rollouts=parameter_rollouts, norm_grad=False,
                        discrete_agent=discrete_agent, print_freq=100)
    agent.discrete_agent.gamma_step_adjust = True

    if initialise_params:
        for a in range(env.action_space.spaces[0].n):
            agent.parameter_weights[a][0,0] = initial_params[a]

    max_steps = 201
    start_time = time.time()
    agent.learn(env, episodes, max_steps)
    end_time = time.time()
    print("Training took %.2f seconds" % (end_time - start_time))
    env.close()

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
    np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.variances = 0
        agent.discrete_agent.epsilon = 0.
        agent.discrete_agent.temperature = 0.
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        print("Ave. evaluation prob. =", sum(evaluation_returns == 50.) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()
