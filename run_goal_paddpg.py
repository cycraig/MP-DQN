import os
import click
import time
import numpy as np
import gym
import gym_goal
from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH
from gym.wrappers import Monitor
from agents.paddpg import PADDPGAgent
from common import ClickPythonLiteralOption
from common.wrappers import ScaledParameterisedActionWrapper
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper
from common.wrappers import ScaledStateWrapper


def pad_action(act, act_param):
    params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
    params[act] = act_param
    return (act, params)


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
            act, act_param, _, _ = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    return np.array(returns)


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.95, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=128, help='Number of transitions required to start learning.',
              type=int)
@click.option('--use-ornstein-noise', default=True,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=20000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-critic', default=0.01, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor', default=0.0001, help='Soft target network update averaging factor.', type=float)
@click.option('--learning-rate-critic', default=1e-6, help="Critic network learning rate.", type=float)
@click.option('--learning-rate-actor', default=1e-7, help="Actor network learning rate.", type=float)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--reward-scale', default=1. / 50., help="Reward scaling factor.", type=float)
@click.option('--clip-grad', default=1., help="Parameter gradient clipping limit.", type=float)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--save-dir', default="results/goal", help='Model save directory.', type=str)
@click.option('--layers', default="(128,64,)", help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--title', default="PADDPG", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, scale_actions, epsilon_steps, epsilon_final, tau_actor, tau_critic, use_ornstein_noise,
        learning_rate_actor, learning_rate_critic, reward_scale, clip_grad, initialise_params, layers, save_dir,
        title):
    env = gym.make('Goal-v0')
    env = GoalObservationWrapper(env)

    if scale_actions:
        kickto_weights = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                                   [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
        shoot_goal_left_weights = np.array([0.857346647646219686, 0])
        shoot_goal_right_weights = np.array([-0.857346647646219686, 0])
    else:
        xfear = 50.0 / PITCH_LENGTH
        yfear = 50.0 / PITCH_WIDTH
        caution = 5.0 / PITCH_WIDTH
        kickto_weights = np.array([[2.5, 1, 0, xfear, 0], [0, 0, 1 - caution, 0, yfear]])
        shoot_goal_left_weights = np.array([GOAL_WIDTH / 2 - 1, 0])
        shoot_goal_right_weights = np.array([-GOAL_WIDTH / 2 + 1, 0])

    initial_weights = np.zeros((4, 17))
    initial_weights[0, [10, 11, 14, 15]] = kickto_weights[0, 1:]
    initial_weights[1, [10, 11, 14, 15]] = kickto_weights[1, 1:]
    initial_weights[2, 16] = shoot_goal_left_weights[1]
    initial_weights[3, 16] = shoot_goal_right_weights[1]

    initial_bias = np.zeros((4,))
    initial_bias[0] = kickto_weights[0, 0]
    initial_bias[1] = kickto_weights[1, 0]
    initial_bias[2] = shoot_goal_left_weights[0]
    initial_bias[3] = shoot_goal_right_weights[0]

    env = GoalFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)
    env = ScaledStateWrapper(env)
    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    print(env.action_space)
    print(env.observation_space)
    env.seed(seed)
    np.random.seed(seed)
    agent = PADDPGAgent(observation_space=env.observation_space.spaces[0],
                        action_space=env.action_space,
                        batch_size=batch_size,
                        learning_rate_actor=learning_rate_actor,
                        learning_rate_critic=learning_rate_critic,
                        epsilon_steps=epsilon_steps,
                        epsilon_final=epsilon_final,
                        gamma=gamma,
                        clip_grad=clip_grad,
                        tau_actor=tau_actor,
                        tau_critic=tau_critic,
                        initial_memory_threshold=initial_memory_threshold,
                        use_ornstein_noise=use_ornstein_noise,
                        replay_memory_size=replay_memory_size,
                        inverting_gradients=inverting_gradients,
                        n_step_returns=False,
                        adam_betas=(0.9, 0.999),
                        critic_kwargs={'hidden_layers': layers, 'init_type': "kaiming"},
                        actor_kwargs={'hidden_layers': layers, 'init_type': "kaiming",  # 'init_std': 1e-5,  # 0.0001,
                                      'squashing_function': False},
                        seed=seed)

    if initialise_params:
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)
    max_steps = 150
    total_reward = 0.
    returns = []
    start_time = time.time()
    for i in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_actions, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)

        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_actions, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)

            r = reward * reward_scale
            agent.step(state, (act, act_param, all_actions, all_action_parameters), r, next_state,
                       (next_act, next_act_param, next_all_actions, next_all_action_parameters), terminal,
                       optimise=True)

            act, act_param, all_actions, all_action_parameters = next_act, next_act_param, next_all_actions, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward

            if terminal:
                break
        agent.end_episode()

        returns.append(episode_reward)
        total_reward += episode_reward
        if (i + 1) % 100 == 0:
            print('{0:5s} R:{1:.5f} P(S):{2:.4f}'.format(str(i + 1), total_reward / (i + 1),
                                                         (np.array(returns) == 50.).sum() / len(returns)))
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    print(agent)

    returns = env.get_episode_rewards()
    np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)

    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon_final = 0.
        agent.epsilon = 0.
        agent.noise = None
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        print("Ave. evaluation prob. =", sum(evaluation_returns == 50.) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()
