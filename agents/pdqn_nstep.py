from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from agents.memory.memory import MemoryNStepReturns
from agents.pdqn import PDQNAgent
from agents.utils import soft_update_target_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QActorNonDueling(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=None, output_layer_init_std=None,
                 activation="leaky_relu", squashing_function=False, action_input_layer=0, init_type="kaiming", init_std=None):
        super(QActorNonDueling, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # initialise layers
        self.layers = nn.ModuleList()
        lastHiddenLayerSize = self.state_size + self.action_parameter_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(self.state_size + self.action_parameter_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))
        # initialise layers
        for i in range(0, len(self.layers)-1):
            nn.init.kaiming_normal_(self.layers[i].weight.data, nonlinearity=self.activation)
            nn.init.zeros_(self.layers[i].bias.data)

        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight.data, mean=0., std=output_layer_init_std)
        else:
            nn.init.zeros_(self.layers[-1].weight.data)
        nn.init.zeros_(self.layers[-1].bias.data)

    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.01

        x = torch.cat((state, action_parameters), dim=1)
        num_hidden_layers = len(self.layers)-1
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        Q = self.layers[-1](x)
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=None, squashing_function=False,
                 activation="leaky_relu", output_layer_init_std=None, init_type="kaiming", init_std=None):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported for now

        # create layers
        self.layers = nn.ModuleList()
        lastHiddenLayerSize = self.state_size
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(self.state_size, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_parameter_size))
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layers
        for i in range(0, len(self.layers)-1):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        else:
            nn.init.zeros_(self.layers[-1].weight)
            # nn.init.normal_(self.layers[-1].weight.data, eps)
        nn.init.zeros_(self.layers[-1].bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers-1):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        action_params = self.layers[num_hidden_layers-1](x)
        action_params += self.action_parameters_passthrough_layer(state)

        if self.squashing_function:
            assert False  # scaling not implemented yet
            action_params = action_params.tanh()
            action_params = action_params *self.action_param_lim

        return action_params


class PDQNNStepAgent(PDQNAgent):
    """
    P-DQN agent using mixed n-step return targets
    """

    NAME = "P-DQN N-Step Agent"

    def __init__(self,
                 *args,
                 beta=0.5,
                 **kwargs):
        super().__init__(*args, actor_class=QActorNonDueling, actor_param_class=ParamActor, **kwargs)
        self.beta = beta
        assert (self.weighted ^ self.average ^ self.random_weighted) or not (
                self.weighted or self.average or self.random_weighted)
        self.replay_memory = MemoryNStepReturns(self.replay_memory_size, self.observation_space.shape,
                                                (1+self.action_parameter_size,),
                                                next_actions=False, n_step_returns=True)

    def __str__(self):
        desc = super().__str__()
        desc += "Beta: {}\n".format(self.beta)
        return desc

    def _add_sample(self, state, action, reward, next_state, terminal, n_step_return=None):
        assert len(action) == 1 + self.action_parameter_size
        assert n_step_return is not None
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal, n_step_return=n_step_return)

    def _optimize_td_loss(self):
        if self.replay_memory.nb_entries < self.batch_size or \
                self.replay_memory.nb_entries < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals, n_step_returns = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)

        states = torch.from_numpy(states).to(device)
        actions_combined = torch.from_numpy(actions).to(device)  # make sure to separate actions and action-parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(device).squeeze()
        next_states = torch.from_numpy(next_states).to(device)
        terminals = torch.from_numpy(terminals).to(device).squeeze()
        n_step_returns = torch.from_numpy(n_step_returns).to(device)

        # ---------------------- optimise critic ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # compute TD error
            off_policy_target = rewards + (1 - terminals) * self.gamma * Qprime
            on_policy_target = n_step_returns.squeeze()
            target = self.beta * on_policy_target + (1. - self.beta) * off_policy_target

        # compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimise actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        assert (self.weighted ^ self.average ^ self.random_weighted) or \
                not (self.weighted or self.average or self.random_weighted)
        Q = self.actor(states, action_params)
        Q_val = Q
        if self.weighted:
            # approximate categorical probability density (i.e. counting)
            counts = Counter(actions.cpu().numpy())
            weights = torch.from_numpy(
                np.array([counts[a] / actions.shape[0] for a in range(self.num_actions)])).float().to(self.device)
            Q_val = weights * Q
        elif self.average:
            Q_val = Q / self.num_actions
        elif self.random_weighted:
            weights = np.random.uniform(0, 1., self.num_actions)
            weights /= np.linalg.norm(weights)
            weights = torch.from_numpy(weights).float().to(self.device)
            Q_val = weights * Q
        if self.indexed:
            Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.actor.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
