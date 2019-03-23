import click
import time
import gym
import os
import numpy as np
import gym_goal
from agents.qpamdp import QPAMDPAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from common.wrappers import ScaledStateWrapper, QPAMDPScaledParameterisedActionWrapper
from gym_goal.envs.config import GOAL_WIDTH, PITCH_WIDTH, PITCH_LENGTH
from gym.wrappers import Monitor
from common.goal_domain import CustomFourierBasis, GoalObservationWrapper

variances = [0.01, 0.01, 0.01]
xfear = 50.0 / PITCH_LENGTH
yfear = 50.0 / PITCH_WIDTH
caution = 5.0 / PITCH_WIDTH
kickto_weights = np.array([[2.5, 1, 0, xfear, 0], [0, 0, 1 - caution, 0, yfear]])
initial_parameter_weights = [
    kickto_weights,
    np.array([[GOAL_WIDTH / 2 - 1, 0]]),
    np.array([[-GOAL_WIDTH / 2 + 1, 0]])
]


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
@click.option('--scale', default=False, help='Scale inputs and actions.', type=bool)  # default 50, 25 best
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--save-dir', default="results/goal", help='Output directory.', type=str)
@click.option('--title', default="QPAMDP", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, scale, initialise_params, save_dir, title):
    alpha_param = 0.1

    env = gym.make('Goal-v0')
    env = GoalObservationWrapper(env)
    if scale:
        variances[0] = 0.0001
        variances[1] = 0.0001
        variances[2] = 0.0001
        alpha_param = 0.06
        initial_parameter_weights[0] = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                                   [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
        initial_parameter_weights[1] = np.array([0.857346647646219686, 0])
        initial_parameter_weights[2] = np.array([-0.857346647646219686, 0])
        env = ScaledStateWrapper(env)
        env = QPAMDPScaledParameterisedActionWrapper(env)

    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    env.seed(seed)
    np.random.seed(seed)

    action_obs_index = np.arange(14)
    param_obs_index = np.array([
        np.array([10, 11, 14, 15]),  # ball_features
        np.array([16]),  # keeper_features
        np.array([16]),  # keeper_features
    ])
    basis = CustomFourierBasis(14, env.observation_space.spaces[0].low[:14], env.observation_space.spaces[0].high[:14])
    discrete_agent = SarsaLambdaAgent(env.observation_space.spaces[0], env.action_space.spaces[0], basis=basis, seed=seed, alpha=0.01,
                                      lmbda=0.1, gamma=0.9, temperature=1.0, cooling=1.0, scale_alpha=False,
                                      use_softmax=True,
                                      observation_index=action_obs_index, gamma_step_adjust=False)
    agent = QPAMDPAgent(env.observation_space.spaces[0], env.action_space, alpha=alpha_param, initial_action_learning_episodes=4000,
                        seed=seed, action_obs_index=action_obs_index, parameter_obs_index=param_obs_index,
                        variances=variances, discrete_agent=discrete_agent, action_relearn_episodes=2000,
                        parameter_updates=1000, parameter_rollouts=50, norm_grad=True, print_freq=100,
                        phi0_func=lambda state: np.array([1, state[1], state[1]**2]),
                        phi0_size=3)
    # Alternating learning periods from original paper:
    # QPAMDP(1) : init(2000), parameter_updates(50), relearn(50)
    # QPAMDP(infinity) : init(2000), parameter_updates(1000), relearn(2000)
    # needed to increase initial action learning episodes to 4000

    if initialise_params:
        for a in range(3):
            agent.parameter_weights[a] = initial_parameter_weights[a]

    max_steps = 150
    start_time = time.time()
    agent.learn(env, episodes, max_steps)
    end_time = time.time()
    print("Training took %.2f seconds" % (end_time - start_time))
    env.close()

    returns = np.array(env.get_episode_rewards())
    print("Saving training results to:",os.path.join(dir, "QPAMDP{}".format(str(seed))))
    np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)

    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
    print('Total P(S):{0:.4f}'.format((returns == 50.).sum() / len(returns)))
    print('Ave. last 100 episode P(S):{0:.4f}'.format((returns[-100:] == 50.).sum() / 100.))

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
