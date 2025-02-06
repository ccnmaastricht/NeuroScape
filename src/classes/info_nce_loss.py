import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):

    def __init__(self, temperature, cutoff_values):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.positive_cutoff, self.negative_cutoff = cutoff_values

    def forward(self, embedding):
        """
        Compute the InfoNCE loss.

        Parameters:
        - embedding: Tensor of shape [num_samples, num_features] containing the embeddings.

        Returns:
        - loss: Float, the loss value.
        """

        # Compute cosine similarities of the new embeddings
        similarity_matrix = torch.matmul(embedding, embedding.t())

        positive_logsum = torch.logsumexp(
            similarity_matrix * self.positives_mask.float() / self.temperature,
            dim=1)
        negative_logsum = torch.logsumexp(
            similarity_matrix * self.negatives_mask.float() / self.temperature,
            dim=1)
        losses = -positive_logsum + negative_logsum
        return losses.mean()

    def get_masks(self, X):
        """
        Get positive and negative masks.

        Parameters:
        - X: Tensor of shape [num_samples, num_features] containing the original embeddings.

        Returns:
        - positives_mask: Tensor of shape [num_samples, num_samples] containing the positive mask.
        - negatives_mask: Tensor of shape [num_samples, num_samples] containing the negative mask.
        """
        # Compute cosine similarities of the original embeddings
        similarity_matrix = torch.matmul(X, X.t())
        # Get positive and negative masks
        positives_mask = similarity_matrix >= self.positive_cutoff
        self.negatives_mask = similarity_matrix <= self.negative_cutoff
        # Ensure self is not considered as positive (diagonal masking)
        self.positives_mask = positives_mask & ~torch.eye(
            similarity_matrix.size(0),
            dtype=torch.bool,
            device=similarity_matrix.device)
