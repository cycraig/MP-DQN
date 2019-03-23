import numpy as np
import warnings

from agents.agent import Agent
from agents.basis import SimpleBasis, ScaledBasis, PolynomialBasis
from agents.sarsa_lambda import SarsaLambdaAgent


class QPAMDPAgent(Agent):
    """
    Defines an agent to optimize H(theta) using the episodic natural actor critic (eNAC) algorithm for continuous
    action spaces.

    Uses Gaussian policy for continuous actions.

    N.B. assumes same state variables used for all actions, and separately same for all parameters
    """
    name = "Q-PAMDP"

    def __init__(self, observation_space, action_space,
                 alpha=0.01,
                 initial_action_learning_episodes=10000,
                 action_relearn_episodes=1000,
                 parameter_updates=180,
                 parameter_rollouts=50,
                 action_obs_index=None,
                 parameter_obs_index=None,
                 discrete_agent=None,
                 norm_grad=False,
                 variances=None,  # list of variances per continuous action parameter (one entry per action)
                 seed=None,
                 phi0_func=None,
                 phi0_size=None,
                 poly_basis=False,
                 print_freq=1):
        super().__init__(observation_space, action_space)

        # split the action space into the discrete actions and continuous parameters
        self.discrete_action_space = action_space.spaces[0]
        self.parameter_space = action_space.spaces[1]
        self.num_actions = self.discrete_action_space.n
        nvars = self.observation_space.shape[0]

        self.alpha = alpha
        if isinstance(variances, (list, np.ndarray)):
            assert len(variances) == self.num_actions
        else:
            variances = variances*np.ones((self.num_actions,))
        self.variances = variances
        self.initial_action_learning_episodes = initial_action_learning_episodes
        self.action_relearn_episodes = action_relearn_episodes
        self.parameter_updates = parameter_updates
        self.parameter_rollouts = parameter_rollouts
        self.episodes_per_cycle = self.action_relearn_episodes + self.parameter_updates * self.parameter_rollouts
        self.parameter_obs_index = parameter_obs_index
        self.norm_grad = norm_grad

        self.phi0_func = phi0_func
        self.phi0_size = phi0_size
        if self.phi0_size is None: assert self.phi0_func is None  # raise error? Need to specify size of custom phi0_func

        self.print_freq = print_freq
        self.R = 0.
        self._total_episodes = 0

        # initialise discrete action learner
        self.discrete_agent = discrete_agent
        if self.discrete_agent is None:
            self.discrete_agent = SarsaLambdaAgent(self.observation_space, self.discrete_action_space, alpha=1.0,
                                                   gamma=0.999, temperature=1.0, cooling=0.995, lmbda=0.5, order=6,
                                                   scale_alpha=True, use_softmax=True, seed=seed,
                                                   observation_index=action_obs_index)

        self.np_random = None
        self.__seed = 0
        self._seed(seed)

        # initialise basis for each action-parameter (one per action)
        if self.parameter_obs_index is not None:
            self.basis = []
            if isinstance(self.parameter_obs_index[0], (list, np.ndarray)):
                if len(self.parameter_obs_index) == 1:
                    self.parameter_obs_index = np.tile(self.parameter_obs_index, (self.num_actions, 1))
                else:
                    # different observation variables for each action-parameter
                    assert len(self.parameter_obs_index) == self.num_actions
            else:
                assert isinstance(self.parameter_obs_index[0], int)
                # same observation variables for all action-parameters, duplicate them for convenience0
                self.parameter_obs_index = np.tile(self.parameter_obs_index,(self.num_actions,1))

            for a in range(self.num_actions):
                nvars = len(self.parameter_obs_index[a])
                low = self.observation_space.low[self.parameter_obs_index[a]]
                high = self.observation_space.high[self.parameter_obs_index[a]]
                # self.basis.append(ScaledBasis(nvars, low, high, bias_unit=True))

                if poly_basis is True:
                    self.basis.append(PolynomialBasis(nvars, order=2, bias_unit=True))
                else:
                    self.basis.append(SimpleBasis(nvars, bias_unit=True))
                # self.basis.append(SimpleBasis(nvars, bias_unit=True))
        else:
            # use simple basis with bias unit (for parameter initialisation)
            # self.basis = [ScaledBasis(nvars, low, high, bias_unit=True) for _ in range(self.num_actions)]
            # if poly_basis is True:
            #     self.basis = [PolynomialBasis(nvars, order=2, bias_unit=True) for _ in range(self.num_actions)]
            # else:
            #     self.basis = [SimpleBasis(nvars, bias_unit=True) for _ in range(self.num_actions)]
            self.basis = [SimpleBasis(nvars, bias_unit=True) for _ in range(self.num_actions)]
        self.num_basis_functions = [self.basis[a].get_num_basis_functions() for a in range(self.num_actions)]
        # self.poly_basis = poly_basis

        # self.parameter_weights = np.zeros((self.num_actions, self.num_basis_functions))  # TODO: randomly init weights?
        # for multidimensional parameters
        self.parameter_weights = []
        for a in range(self.num_actions):
            shape = (self.num_basis_functions[a],)
            param_shape = self.parameter_space.spaces[a].shape
            assert len(param_shape) <= 1
            if len(param_shape) == 1 and param_shape[0] > 0:
                shape = (param_shape[0], self.num_basis_functions[a])
            self.parameter_weights.append(np.zeros(shape))
            # self.parameter_weights.append(self.np_random.normal(loc=0.,scale=0.0001,size=shape))
        # self.parameter_weights = self.np_random.random_sample((self.num_actions, self.num_basis_functions))

    def act(self, state):
        act = self._action_policy(state)
        param = self._parameter_policy(state, act)
        return self._pad_action(act, param)

    def learn(self, env, max_episodes=100000, max_steps_per_episode=None):
        """ Learn for a given number of episodes. """
        self.e = 0
        if max_episodes < self.initial_action_learning_episodes:
            warnings.warn("Too few episodes to initialise agent!", UserWarning)

        print("Initial discrete action learning for %d episodes..." % self.initial_action_learning_episodes)
        for _ in range(self.initial_action_learning_episodes):
            self._rollout(env, update_actions=True, max_steps=max_steps_per_episode)
            self.e += 1
            if self.e > max_episodes: break

        while True:
            self.discrete_agent.temperature = 0.0
            self.discrete_agent.epsilon = 0.0

            # update parameter policy
            print(self.e, "Updating parameter selection...")
            for _ in range(self.parameter_updates):
                self._parameter_update(env, max_steps_per_episode)
                self.e += self.parameter_rollouts
                if self.e > max_episodes: break
            if self.e > max_episodes: break

            self.discrete_agent.temperature = 1.0
            self.discrete_agent.epsilon = 1.0

            # update discrete action policy
            print(self.e, "Updating action selection...")
            for _ in range(self.action_relearn_episodes):
                self._rollout(env, update_actions=True, max_steps=max_steps_per_episode)
                self.e += 1
                if self.e > max_episodes: break
            if self.e > max_episodes: break

        # no stochastic actions for evaluation?
        self.discrete_agent.temperature = 0.0
        self.discrete_agent.epsilon = 0.0

    def start_episode(self):
        self.discrete_agent.start_episode()

    def end_episode(self):
        self.discrete_agent.end_episode()

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.np_random = np.random.RandomState(seed=seed)

    def _get_parameters(self):
        """ Returns all the parameters in a vector. """
        # parameters = []
        # # for non-uniform parameter wieghts shapes (ragged array)
        # for a in range(self.num_actions):
        #    parameters.append(self.parameter_weights[a])
        # return np.ravel(self.parameter_weights)  # np.array(parameters)
        return np.concatenate([self.parameter_weights[i].flat for i in range(len(self.parameter_weights))])

    def _set_parameters(self, parameters):
        """ Set the parameters using a vector. """
        index = 0
        for action in range(self.num_actions):
            rows = self.parameter_weights[action].size
            self.parameter_weights[action] = parameters[index: index + rows].reshape(self.parameter_weights[action].shape)
            index += rows

    def _log_parameter_gradient(self, state, act, param):
        """ Returns the log gradient for the parameter,
            given the state and the value. """
        features = self._compute_features(state, act)
        mean = self.parameter_weights[act].dot(features)
        grad = np.outer((param - mean),features / self.variances[act])
        return grad.ravel()

    def log_gradient(self, state, action, param):
        """ Returns the log gradient for the entire policy. """
        grad = np.zeros((0,))
        for i in range(self.num_actions):
            elems = self.parameter_weights[i].size
            if i == action:
                parameter_grad = self._log_parameter_gradient(state, i, param)
                grad = np.append(grad, parameter_grad)
            else:
                grad = np.append(grad, np.zeros((elems,)))
        return grad

    def _pad_action(self, act, param):
        # Box for each parameter wrapped in a Compound
        action = [np.zeros(self.parameter_space.spaces[a].shape) for a in range(self.num_actions)]
        action[act] = param
        action = (act, action)
        return action

    def _rollout(self, env, update_actions=False, max_steps=None):
        """ Run a single episode for a maximum number of steps. """
        state, _ = env.reset()
        states = [state]
        rewards = []
        actions = []
        terminal = False
        act = self._action_policy(state)
        acts = [act]

        steps = 0
        if update_actions:
            self.discrete_agent.start_episode()
        while not terminal and not (max_steps is not None and steps > max_steps):
            param = self._parameter_policy(state, act)
            # print (act,param)
            (new_state, time_steps), reward, terminal, _ = env.step(self._pad_action(act, param))
            new_act = self._action_policy(new_state)

            if update_actions:
                self.discrete_agent.step(state, act, reward, new_state, new_act, terminal, time_steps)
            state = new_state
            states.append(state)
            actions.append((act, param))
            rewards.append(reward)
            act = new_act
            acts.append(act)

            steps += 1
        if update_actions:
            self.discrete_agent.end_episode()

        self.R += sum(rewards)
        self._total_episodes += 1
        if self.print_freq > 0 and self._total_episodes % self.print_freq == 0:
            if self.print_freq == 1:
                print("{0:5s} R: {1:.4f} r: {2:.4f}".format(str(self._total_episodes), self.R/self._total_episodes,sum(rewards)))
            else:
                # print("{0:5s} R: {1:.4f}".format(str(self._total_episodes), self.R/self._total_episodes))
                returns = np.array(env.get_episode_rewards())
                print('{0:5s} R:{1:.5f} P(S):{2:.4f}'.format(str(self._total_episodes), sum(returns) / (self._total_episodes),
                                                             (np.array(returns) == 50.).sum() / len(returns)))

        return states, actions, rewards, acts

    def _enac_gradient(self, env, max_steps=None):  #, phi0_func=None, phi0_size=None):
        """
        Compute the episodic NAC gradient.

        phi0_func : lambda function giving the state features of s_0, the initial state in a trajectory
                    defaults to [1.] if None
        phi0_size : number of features returned by phi0_func
        """
        if self.phi0_size is None: assert self.phi0_func is None  # raise error? Need to specify size of custom phi0_fun
        if self.phi0_func is None:
            self.phi0_func = lambda state: np.array([1,])
            self.phi0_size = 1
        returns = np.zeros((self.parameter_rollouts, 1))
        param_size = self._get_parameters().size
        psi = np.zeros((self.parameter_rollouts, param_size + self.phi0_size))
        for run in range(self.parameter_rollouts):
            states, actions, rewards, acts = self._rollout(env, False, max_steps)
            returns[run, 0] = sum(rewards)
            log_grad = np.zeros((param_size,))
            for state, act, action in zip(states, acts, actions):
                log_grad += self.log_gradient(state, act, action[1])
            psi[run, :] = np.append(log_grad, self.phi0_func(states[0]))
        grad = np.linalg.pinv(psi).dot(returns)[0:param_size, 0]
        return grad

    def _parameter_update(self, env, max_steps=None):
        """ Perform a single gradient update. """
        grad = self._enac_gradient(env, max_steps)
        if np.linalg.norm(grad) > 0 and self.norm_grad:
            grad /= np.linalg.norm(grad)

        self._set_parameters(self._get_parameters() + self.alpha * grad)

    def _action_update(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        self.discrete_agent.step(state, action[0], reward, next_state, next_action[0], terminal, time_steps)

    def _action_policy(self, state):
        return self.discrete_agent.act(state)

    def _parameter_policy(self, state, act):
        return self._gaussian_policy(state, act)

    def _gaussian_policy(self, state, act):
        """ Gaussian action policy for continuous actions. """
        mean = np.dot(self.parameter_weights[act], self._compute_features(state, act))
        variance = 0.
        if self.variances is not None:
            if isinstance(self.variances, (list, np.ndarray)):
                variance = self.variances[act]
            else:
                variance = self.variances

        if variance == 0.:
            return mean
        else:
            # TODO: multivariate_normal expects variance, normal expects stdev? may be important...
            # this may be incorrect / unnecessary but trying to be consistent with Warwick's source code for now
            if isinstance(mean, np.ndarray) and len(mean) > 1:
                return self.np_random.multivariate_normal(mean, variance*np.eye(len(mean)))
            return self.np_random.normal(mean, variance)

    def _compute_features(self, state, act):
        """ Returns phi: the features after the function approximation basis has been applied. """
        if self.parameter_obs_index is not None:
            state = state[self.parameter_obs_index[act]]
        return self.basis[act].compute_features(state)

    def __str__(self):
        desc = ("Q-PAMDP Agent\n"+
                "Alpha: {}\n".format(self.alpha)+
                "Initial Action Episodes: {}\n".format(self.initial_action_learning_episodes)+
                "Action Relearn Episodes: {}\n".format(self.action_relearn_episodes)+
                "Parameter Updates: {}\n".format(self.parameter_updates) +
                "Parameter Rollouts: {}\n".format(self.parameter_rollouts) +
                "Observation Index: {}\n".format(self.parameter_obs_index) +
                "Variances: {}\n".format(self.variances) +
                "Norm Grad: {}\n".format(self.norm_grad) +
                "Phi0 func.: {}\n".format(self.phi0_func) +
                "Phi0 size: {}\n".format(self.phi0_size) +
                "Discrete Agent: {}\n".format(self.discrete_agent) +
                "Seed: {}\n".format(self.__seed))

        return desc
