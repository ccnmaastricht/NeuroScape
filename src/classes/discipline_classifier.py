import torch.nn as nn


class DisciplineClassifier(nn.Module):

    def __init__(self, layer_sizes, num_classes, dropout=0.5):
        super(DisciplineClassifier, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ELU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
        )
        self.dropout = nn.Dropout(dropout)
        self.multiclass = nn.Linear(layer_sizes[2], num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sequential(x)
        x = self.dropout(x)
        x = self.multiclass(x)
        x = self.sigmoid(x)
        return x
