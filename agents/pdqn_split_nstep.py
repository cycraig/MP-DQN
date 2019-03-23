import torch
import torch.optim as optim

from agents.pdqn_nstep import PDQNNStepAgent
from agents.pdqn_split import SplitQActor
from agents.utils import hard_update_target_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PDQNNStepSplitAgent(PDQNNStepAgent):
    NAME = "Split P-DQN N-Step Agent"

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = SplitQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                 **kwargs['actor_kwargs']).to(device)
        self.actor_target = SplitQActor(self.observation_space.shape[0], self.num_actions, self.action_parameter_sizes,
                                        **kwargs['actor_kwargs']).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
