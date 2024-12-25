from torch import nn

from config import NUM_HIDDEN_NEURON


class DQN(nn.Sequential):

    def __init__(self, n_states, n_actions):

        super(DQN, self).__init__(
            nn.Linear(n_states, NUM_HIDDEN_NEURON),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_NEURON, NUM_HIDDEN_NEURON),
            nn.ReLU(),
            nn.Linear(NUM_HIDDEN_NEURON, n_actions)
        )