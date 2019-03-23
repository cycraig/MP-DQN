from agents.agent import Agent


class RandomAgent(Agent):
    """
    Defines an agent that acts uniformly randomly.
    """
    name = "Random"

    def act(self, state):
        return self.action_space.sample()

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        pass

    def start_episode(self):
        pass

    def end_episode(self):
        pass
