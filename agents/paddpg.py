import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from torch.autograd import Variable

from agents.agent import Agent
from agents.memory.memory import Memory, MemoryNStepReturns
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=None, action_input_layer=0,
                 init_type="normal", activation="leaky_relu", init_std=0.01):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # self.action_input_layer = action_input_layer

        # initialise layers
        self.layers = nn.ModuleList()
        input_size = self.state_size + action_size + action_parameter_size
        last_hidden_layer_size = input_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(input_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            last_hidden_layer_size = hidden_layers[nh - 1]
        self.output_layer = nn.Linear(last_hidden_layer_size, 1)

        # initialise layers
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight.data, nonlinearity=self.activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight.data, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias.data)
        nn.init.normal_(self.output_layer.weight, std=init_std)
        # nn.init.normal_(self.action_output_layer.bias, std=init_std)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state, actions, action_parameters):
        x = torch.cat((state, actions, action_parameters), dim=1)
        negative_slope = 0.01

        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        Q = self.output_layer(x)

        return Q


class Actor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=None, init_std=0.01,
                 init_type="normal", activation="leaky_relu", squashing_function=False):
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function=squashing_function
        assert self.squashing_function is False  # unsupported, still need to implement rescaling of output
        self.activation = activation

        # initialise layers
        self.layers = nn.ModuleList()
        last_hidden_layer_size = self.state_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(self.state_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            last_hidden_layer_size = hidden_layers[nh - 1]

        # separate output layers for actions and action-parameters
        self.action_output_layer = nn.Linear(last_hidden_layer_size, self.action_size)
        self.action_parameters_output_layer = nn.Linear(last_hidden_layer_size, self.action_parameter_size)

        # initialise layers
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight.data, nonlinearity=self.activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight.data, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias.data)
        nn.init.normal_(self.action_output_layer.weight, std=init_std)
        nn.init.zeros_(self.action_output_layer.bias)
        nn.init.normal_(self.action_parameters_output_layer.weight, std=init_std)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)
        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        negative_slope = 0.01

        x = state
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        actions = self.action_output_layer(x)
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)

        # if self.squashing_function:
        #     assert False  # scaling not implemented yet
        #     '''
        #     actions = actions.tanh()
        #     actions = actions *self.action_lim
        #     action_params = action_params.tanh()
        #     action_params = action_params *self.action_param_lim
        #     '''

        return actions, action_params


class PADDPGAgent(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=Actor,
                 actor_kwargs={},
                 critic_class=Critic,
                 critic_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.01,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 beta=0.5, # averaging factor between off-policy and on-policy targets during n-step updates
                 tau_actor=0.001,  # Polyak averaging factor for updating target weights
                 tau_critic=0.001,
                 replay_memory=None,  # memory buffer object
                 replay_memory_size=1000000,
                 learning_rate_actor=0.00001,
                 learning_rate_critic=0.001,
                 initial_memory_threshold=0,
                 clip_grad=10,
                 adam_betas=(0.95, 0.999),
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss,  # F.smooth_l1_loss
                 inverting_gradients=False,
                 n_step_returns=False,
                 seed=None):
        super(PADDPGAgent, self).__init__(observation_space, action_space)

        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_actions+1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low for i in range(1,self.num_actions+1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps

        self.clip_grad = clip_grad
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta = beta
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_critic = tau_critic
        self._step = 0
        self._episode = 0
        self.updates = 0

        self.np_random = None
        self.seed = seed
        self._seed(seed)

        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001)

        print(self.num_actions+self.action_parameter_size)
        self.n_step_returns = n_step_returns
        if replay_memory is None:
            self.replay_memory = MemoryNStepReturns(replay_memory_size, observation_space.shape, (self.num_actions+self.action_parameter_size,), next_actions=False, n_step_returns=self.n_step_returns)
        else:
            self.replay_memory = replay_memory
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic = critic_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **critic_kwargs).to(device)
        self.critic_target = critic_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **critic_kwargs).to(device)
        hard_update_target_network(self.critic, self.critic_target)
        self.critic_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor, betas=adam_betas)
        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic, betas=adam_betas)

    def __str__(self):
        desc = ("P-DDPG Agent with frozen initial weight layer\n" +
                "Actor: {}\n".format(self.actor) +
                "Critic: {}\n".format(self.critic) +
                "Actor Alpha: {}\n".format(self.learning_rate_actor) +
                "Critic Alpha: {}\n".format(self.learning_rate_critic) +
                "Gamma: {}\n".format(self.gamma) +
                "Tau Actor: {}\n".format(self.tau_actor) +
                "Tau Critic: {}\n".format(self.tau_critic) +
                "Beta: {}\n".format(self.beta) +
                "Inverting Gradients: {}\n".format(self.inverting_gradients) +
                "Replay Memory: {}\n".format(self.replay_memory_size) +
                "epsilon_initial: {}\n".format(self.epsilon_initial) +
                "epsilon_final: {}\n".format(self.epsilon_final) +
                "epsilon_steps: {}\n".format(self.epsilon_steps) +
                "Clip norm: {}\n".format(self.clip_grad) +
                "Batch Size: {}\n".format(self.batch_size) +
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) +
                "Seed: {}\n".format(self.seed))
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor.action_parameters_passthrough_layer
        print(initial_weights.shape)
        print(passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(device)
        if initial_bias is not None:
            print(initial_bias.shape)
            print(passthrough_layer.bias.data.size())
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor, self.actor_target)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        # 5x faster on CPU
        if grad_type == "actions":
            max_p = self.action_max.cpu()
            min_p = self.action_min.cpu()
            rnge = self.action_range.cpu()
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max.cpu()
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]):
                # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
                index = grad[n] > 0
                grad[n][index] *= (index.float() * (max_p - vals[n]) / rnge)[index]
                grad[n][~index] *= ((~index).float() * (vals[n] - min_p) / rnge)[~index]

        return grad

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        # anneal exploration
        if self._episode < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    self._episode / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final
        pass

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(device)
            all_actions, all_action_parameters = self.actor.forward(state)
            all_actions = all_actions.detach().cpu().data.numpy()
            all_action_parameters = all_action_parameters.detach().cpu().data.numpy()

            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            if self.np_random.uniform() < self.epsilon:
                all_actions = self.np_random.uniform(size=all_actions.shape)
                offsets = np.array([self.action_parameter_sizes[i] for i in range(self.num_actions)], dtype=int).cumsum()
                offsets = np.concatenate((np.array([0]), offsets))
                if not self.use_ornstein_noise:
                    for i in range(self.num_actions):
                        all_action_parameters[offsets[i]:offsets[i+1]] = self.np_random.uniform(
                            self.action_parameter_min_numpy[offsets[i]:offsets[i+1]],
                            self.action_parameter_max_numpy[offsets[i]:offsets[i+1]])

            # select maximum action
            action = np.argmax(all_actions)
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            if self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
            action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]
        return action, action_parameters, all_actions, all_action_parameters

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1, optimise=True):
        action, action_params, all_actions, all_action_parameters = action
        self._step += 1

        self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        if optimise and self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()

    def _add_sample(self, state, action, reward, next_state, terminal):
        assert not self.n_step_returns
        assert len(action) == self.num_actions+self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal)

    def _optimize_td_loss(self):
        if self.replay_memory.nb_entries < self.batch_size or \
                self.replay_memory.nb_entries < self.initial_memory_threshold:
            return

        # Sample a batch from replay memory
        if self.n_step_returns:
            states, actions, rewards, next_states, terminals, n_step_returns = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
        else:
            states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
            n_step_returns = None

        states = torch.from_numpy(states).to(device)
        actions_combined = torch.from_numpy(actions).to(device)  # make sure to separate actions and action-parameters
        actions = actions_combined[:,:self.num_actions]
        action_parameters = actions_combined[:, self.num_actions:]
        rewards = torch.from_numpy(rewards).to(device)
        next_states = torch.from_numpy(next_states).to(device)
        terminals = torch.from_numpy(terminals).to(device)
        if self.n_step_returns:
            n_step_returns = torch.from_numpy(n_step_returns).to(device)

        # ---------------------- optimize critic ----------------------
        with torch.no_grad():
            pred_next_actions, pred_next_action_parameters = self.actor_target.forward(next_states)
            off_policy_next_val = self.critic_target.forward(next_states, pred_next_actions, pred_next_action_parameters)
            off_policy_target = rewards + (1 - terminals) * self.gamma * off_policy_next_val
            if self.n_step_returns:
                on_policy_target = n_step_returns
                target = self.beta*on_policy_target + (1.-self.beta) * off_policy_target
            else:
                target = off_policy_target

        y_expected = target
        y_predicted = self.critic.forward(states, actions, action_parameters)
        loss_critic = self.loss_func(y_predicted, y_expected)

        self.critic_optimiser.zero_grad()
        loss_critic.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimiser.step()

        # ---------------------- optimise actor ----------------------
        # 1 - calculate gradients from critic
        with torch.no_grad():
            actions, action_params = self.actor(states)
            action_params = torch.cat((actions, action_params), dim=1)
        action_params.requires_grad = True
        Q_val = self.critic(states, action_params[:, :self.num_actions], action_params[:, self.num_actions:]).mean()
        self.critic.zero_grad()
        Q_val.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # 2 - apply inverting gradients and combine with gradients from actor
        actions, action_params = self.actor(Variable(states))
        action_params = torch.cat((actions, action_params), dim=1)
        delta_a[:, self.num_actions:] = self._invert_gradients(delta_a[:, self.num_actions:].cpu(), action_params[:, self.num_actions:].cpu(), grad_type="action_parameters", inplace=True)
        delta_a[:, :self.num_actions] = self._invert_gradients(delta_a[:, :self.num_actions].cpu(), action_params[:, :self.num_actions].cpu(), grad_type="actions", inplace=True)
        out = -torch.mul(delta_a, action_params)
        self.actor.zero_grad()
        out.backward(torch.ones(out.shape).to(device))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.critic, self.critic_target, self.tau_critic)
