import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEmbeddingNetwork(nn.Module):

    def __init__(self,
                 input_dimension=1024,
                 hidden_dimensions=[512, 128, 64],
                 output_dimension=32,
                 dropout=0.05):
        super(SparseEmbeddingNetwork, self).__init__()
        self.first_stage = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.BatchNorm1d(hidden_dimensions[0]), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1]), nn.ELU())

        self.second_stage = nn.Sequential(
            nn.Linear(hidden_dimensions[1], hidden_dimensions[2]),
            nn.BatchNorm1d(hidden_dimensions[2]), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimensions[2], output_dimension))

    def forward(self, x):
        """
        Forward pass of the network.
        
        Parameters:
        - x: Tensor, the input tensor.
        
        Returns:
        - embedding: Tensor, the output tensor.
        """
        first_state = self.first_stage(x)
        second_state = self.second_stage(first_state)
        embedding = F.normalize(second_state, p=2, dim=1)

        return embedding
