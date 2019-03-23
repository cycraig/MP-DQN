import numpy as np
from agents.agent import Agent
from agents.basis import FourierBasis


class SarsaLambdaAgent(Agent):
    """
    Sarsa(lambda) agent with Fourier basis function approximation.

    Alpha scaling (Will Dabney) is used to tune the learning rate in a non-increasing manner.
    Only discrete action spaces are supported by this algorithm.
    By default a softmax action selection policy is used; if disabled we fall back to an epsilon-greedy policy.
    """
    name = "Sarsa(lambda)"

    def __init__(self, observation_space, action_space,
                 alpha=0.01,
                 gamma=0.99,
                 epsilon=0.1,
                 temperature=1.0,
                 cooling=0.996,
                 lmbda=0.7,
                 order=3,
                 scale_alpha=True,
                 use_softmax=True,
                 seed=None,
                 basis=None,
                 observation_index=None,  # array-like: indices of the variables to use from the observation space
                 gamma_step_adjust=False
                 ):
        """"

        :param observation_space:
        :param action_space:
        :param alpha:
        :param gamma:
        :param epsilon: random factor for epsilon-greedy exploration (ignored if use_softmax is True)
        :param temperature: random factor for softmax action selection (ignored if use_softmax is False)
        :param cooling: decay factor for temperature
        :param lmbda:
        :param order:
        :param scale_alpha:
        :param use_softmax:
        """
        super().__init__(observation_space, action_space)

        assert np.isfinite(self.action_space.n)
        assert len(self.observation_space.shape) == 1  # only 1D observation spaces supported

        self.epsilon = epsilon
        self.temperature = temperature
        self.cooling = cooling
        self.alpha = alpha
        self.lmbda = lmbda
        self.gamma = gamma
        self.scale_alpha = scale_alpha
        self.use_softmax = use_softmax
        self.observation_index = observation_index
        self.gamma_step_adjust = gamma_step_adjust

        nvars = self.observation_space.shape[0]
        low = self.observation_space.low
        high = self.observation_space.high
        if self.observation_index is not None:
            nvars = len(observation_index)
            low = low[observation_index]
            high = high[observation_index]
        if basis is None:
            self.basis = FourierBasis(nvars, low, high, order)
        else:
            self.basis = basis
        self.num_actions = self.action_space.n
        self.num_basis_functions = self.basis.get_num_basis_functions()

        self.np_random = None
        self.__seed = None
        self._seed(seed)

        self.weights = np.zeros((self.num_actions, self.num_basis_functions))
        # self.weights = self.np_random.random_sample((self.num_actions, self.num_basis_functions))
        self.traces = np.zeros(self.weights.shape)

    def act(self, state):
        if self.use_softmax:
            return self._softmax_policy(state)
        else:
            return self._epsilon_greedy_policy(state)

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        phi = self._compute_features(state)
        next_phi = self._compute_features(next_state)
        shrink = self.basis.get_shrink()

        # update traces
        self.traces *= self.lmbda * self.gamma
        self.traces[action] += phi

        # adjust gamma to account for multiple time steps between actions; use with caution
        gamma = self.gamma ** time_steps if self.gamma_step_adjust else self.gamma

        # alpha scaling
        if self.scale_alpha:
            alpha_bound = -self.traces[action].dot(phi / shrink)
            if not terminal:
                alpha_bound += gamma * self.traces[next_action].dot(next_phi / shrink)

            if alpha_bound < 0.0:
                self.alpha = min(self.alpha, 1.0 / abs(alpha_bound))

        delta = reward - self.weights[action].dot(phi)
        if not terminal:
            delta += gamma * self.weights[next_action].dot(next_phi)

        self.weights += self.alpha * delta * self.traces / shrink

    def start_episode(self):
        self.traces[:] = 0.

    def end_episode(self):
        # TODO: decay exploration (temperature parameter) outside of class instead?
        if self.use_softmax:
            self.temperature *= self.cooling
        else:
            self.epsilon *= self.cooling

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.__seed = seed
        self.np_random = np.random.RandomState(seed=seed)

    def _epsilon_greedy_policy(self, state):
        """ Action selection with epsilon-greedy exploration. """
        if self.np_random.uniform() < self.epsilon:
            return self.np_random.choice(self.num_actions)  # action_space.sample() is on its own seed, cannot use

        Q = np.dot(self.weights, self._compute_features(state))
        return Q.argmax()

    def _softmax_policy(self, state):
        """" Softmax action selection """
        Q = np.dot(self.weights, self._compute_features(state))
        if self.temperature == 0. or self.temperature < 1e-16:
            return Q.argmax()
        else:
            Q /= self.temperature
            Q -= Q.max()
            Q = np.exp(Q)
            Q /= Q.sum()
            # weighted selection
            rand = self.np_random.random_sample()
            for i, value in enumerate(Q):
                if rand < value:
                    return i
                rand -= value  # or could use Q = Q.cumsum() and remove this?

    def _compute_features(self, state):
        """ Returns phi: the features after the function approximation basis has been applied. """
        if self.observation_index is not None:
            state = state[self.observation_index]
        return self.basis.compute_features(state)

    def __str__(self):
        desc = ("SARSA(lambda) Agent\n"+
                "Alpha: {}\n".format(self.alpha)+
                "Lambda: {}\n".format(self.lmbda)+
                "Gamma: {}\n".format(self.gamma) +
                "Scale Alpha: {}\n".format(self.scale_alpha)+
                "Basis: {}\n".format(self.basis) +
                "Gamma Step Adjust: {}\n".format(self.gamma_step_adjust) +
                "Observation Index: {}\n".format(self.observation_index) +
                "Use Softmax: {}\n".format(self.use_softmax) +
                ("Temperature: {}\n".format(self.temperature) +
                 "Cooling: {}\n".format(self.cooling)) if self.use_softmax else
                ("Epsilon: {}\n".format(self.epsilon)) +
                "Seed: {}\n".format(self.__seed))
        return desc
