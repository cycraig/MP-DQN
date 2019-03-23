import os
import numpy as np
import click
import time
import gym
import gym_soccer
from agents.qpamdp import QPAMDPAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from common.soccer_domain import SoccerParameterisedActionWrapper, SoccerScaledParameterisedActionWrapper
from common.wrappers import TimestepWrapper, ScaledStateWrapper
from gym.wrappers import Monitor


def evaluate(env, agent, episodes=10):
    returns = []
    timesteps = []
    goals = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        info = {'status': "NOT_SET"}
        while not terminal:
            t += 1
            s = np.array(state, dtype=np.float32)
            action = agent.act(s)
            (s, _), reward, terminal, info = env.step(action)
            total_reward += reward
        # print(info['status'])
        goal = info['status'] == 'GOAL'
        timesteps.append(t)
        returns.append(total_reward)
        goals.append(goal)
    return np.column_stack((returns, timesteps, goals))


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--parameter-updates', default=50, help='parameter_updates.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--learning-rate-actor-param', default=0.2, help='eNAC learning rate.', type=float)
@click.option('--learning-rate-actor', default=1.0, help='Sarsa learning rate.', type=float)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--variance', default=0.01, help="Exploration variance.", type=float)
@click.option('--title', default="QPAMDP", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, parameter_updates, gamma, scale_actions, learning_rate_actor,
        learning_rate_actor_param, variance, title):
    env = gym.make('SoccerScoreGoal-v0')
    if scale_actions:
        env = SoccerScaledParameterisedActionWrapper(env)
    env = SoccerParameterisedActionWrapper(env)
    env = TimestepWrapper(env)
    # env = ScaledStateWrapper(env)
    dir = os.path.join(*("results", "soccer", title))
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    # env.seed(seed)
    np.random.seed(seed)
    action_obs_index = [5, 6, 7, 12, 13, 14, 15, 51, 52, 53]
    parameter_obs_index = action_obs_index
    print(env.action_space.spaces[0])
    print(env.observation_space)

    discrete_agent = SarsaLambdaAgent(env.observation_space, env.action_space.spaces[0], seed=seed, alpha=learning_rate_actor,
                                      lmbda=0.5, gamma=gamma, epsilon=1.0, temperature=1.0, observation_index=action_obs_index,
                                      cooling=0.995, scale_alpha=True, use_softmax=False, gamma_step_adjust=False, order=2)
    agent = QPAMDPAgent(env.observation_space, env.action_space, alpha=learning_rate_actor_param, initial_action_learning_episodes=1000,
                        seed=seed, variances=variance, discrete_agent=discrete_agent, action_relearn_episodes=1000,
                        parameter_updates=parameter_updates, parameter_rollouts=25, norm_grad=False,
                        action_obs_index=action_obs_index, parameter_obs_index=parameter_obs_index,
                        print_freq=100, poly_basis=False,
                        #phi0_func=lambda state: np.array([1, state[1], state[1] ** 2]),
                        #phi0_size=3,
                        )
    agent.parameter_weights[0][0, 0] = 0.5
    agent.parameter_weights[2][0, 0] = 0.5
    print(agent)

    agent.learn(env, max_episodes=episodes)

    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
    np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.variances = 0
        agent.discrete_agent.epsilon = 0.
        agent.discrete_agent.temperature = 0.
        start_time_eval = time.time()
        evaluation_results = evaluate(env, agent, evaluation_episodes)  # returns, timesteps, goals
        end_time_eval = time.time()
        print("Ave. evaluation return =", sum(evaluation_results[:, 0]) / evaluation_results.shape[0])
        print("Ave. timesteps =", sum(evaluation_results[:, 1]) / evaluation_results.shape[0])
        goal_timesteps = evaluation_results[:, 1][evaluation_results[:, 2] == 1]
        if len(goal_timesteps) > 0:
            print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
        else:
            print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
        print("Ave. goal prob. =", sum(evaluation_results[:, 2]) / evaluation_results.shape[0])
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_results)
        print("Evaluation time: %.2f seconds" % (end_time_eval - start_time_eval))


if __name__ == '__main__':
    run()
