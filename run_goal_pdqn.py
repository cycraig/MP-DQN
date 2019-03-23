import os
import click
import time
import numpy as np
import gym
import gym_goal
from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.wrappers import ScaledParameterisedActionWrapper
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper
from common.wrappers import ScaledStateWrapper
from agents.pdqn import PDQNAgent
from agents.pdqn_split import SplitPDQNAgent
from agents.pdqn_multipass import MultiPassPDQNAgent


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
            act, act_param, all_action_parameters = agent.act(state)
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
@click.option('--replay-memory-size', default=20000, help='Replay memory transition capacity.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--reward-scale', default=1./50., help="Reward scaling factor.", type=float)
@click.option('--clip-grad', default=1., help="Parameter gradient clipping limit.", type=float)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--layers', default="(256,)", help='Hidden layers.', cls=ClickPythonLiteralOption)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/goal", help='Output directory.', type=str)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="PDQN", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, epsilon_final, tau_actor, tau_actor_param, use_ornstein_noise,
        learning_rate_actor, learning_rate_actor_param, reward_scale, clip_grad, title, scale_actions,
        zero_index_gradients, split, layers, multipass, indexed, weighted, average, random_weighted, render_freq,
        action_input_layer, initialise_params, save_freq, save_dir, save_frames, visualise):

    env = gym.make('Goal-v0')
    env = GoalObservationWrapper(env)

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)
    assert not (save_frames and visualise)
    if visualise:
        assert render_freq > 0
    if save_frames:
        assert render_freq > 0
        vidir = os.path.join(save_dir, "frames")
        os.makedirs(vidir, exist_ok=True)

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

    if not scale_actions:
        # rescale initial action-parameters for a scaled state space
        for a in range(env.action_space.spaces[0].n):
            mid = (env.observation_space.spaces[0].high + env.observation_space.spaces[0].low) / 2.
            initial_bias[a] += np.sum(initial_weights[a] * mid)
            initial_weights[a] = initial_weights[a]*env.observation_space.spaces[0].high - initial_weights[a] * mid

    env = GoalFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)
    env = ScaledStateWrapper(env)
    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    env.seed(seed)
    np.random.seed(seed)

    assert not (split and multipass)
    agent_class = PDQNAgent
    if split:
        agent_class = SplitPDQNAgent
    elif multipass:
        agent_class = MultiPassPDQNAgent
    agent = agent_class(
                       observation_space=env.observation_space.spaces[0], action_space=env.action_space,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,  # 0.0001
                       learning_rate_actor_param=learning_rate_actor_param,  # 0.001
                       epsilon_steps=epsilon_steps,
                       epsilon_final=epsilon_final,
                       gamma=gamma,
                       clip_grad=clip_grad,
                       indexed=indexed,
                       average=average,
                       random_weighted=random_weighted,
                       tau_actor=tau_actor,
                       weighted=weighted,
                       tau_actor_param=tau_actor_param,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       inverting_gradients=inverting_gradients,
                       actor_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-5,
                                     'action_input_layer': action_input_layer,},
                       actor_param_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-5,
                                           'squashing_function': False},
                       zero_index_gradients=zero_index_gradients,
                       seed=seed)

    if initialise_params:
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
    print(agent)
    max_steps = 150
    total_reward = 0.
    returns = []
    start_time = time.time()
    video_index = 0
    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))

        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)

        if visualise and i % render_freq == 0:
            env.render()

        episode_reward = 0.
        agent.start_episode()
        for j in range(max_steps):
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            r = reward * reward_scale
            agent.step(state, (act, all_action_parameters), r, next_state,
                       (next_act, next_all_action_parameters), terminal, steps)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward

            if visualise and i % render_freq == 0:
                env.render()

            if terminal:
                break
        agent.end_episode()

        if save_frames:
            video_index = env.unwrapped.save_render_states(vidir, title, video_index)

        returns.append(episode_reward)
        total_reward += episode_reward
        if (i + 1) % 100 == 0:
            print('{0:5s} R:{1:.5f} P(S):{2:.4f}'.format(str(i + 1), total_reward / (i + 1),
                                                         (np.array(returns) == 50.).sum() / len(returns)))
    end_time = time.time()
    print("Training took %.2f seconds" % (end_time - start_time))
    env.close()

    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

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
