import os
import glob
import torch
import tomllib

import pandas as pd

from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

from src.utils.grokfast import gradfilter_ema


def load_configurations():
    """
    Load the configuration from the config file.
    
    Returns:
    - configurations: dict
    """
    with open('config/domain_embedding/model_training.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def load_log(log_file):
    """
    Load the losses from a file.

    Parameters:
    - log_file: String, the path to the log file.

    Returns:
    - logging: dict
    """
    df = pd.read_csv(log_file)
    logging = df.to_dict(orient='list')

    return logging


def save_model(model, path, name='best_model.pth'):
    """
    Save the model to a file.

    Parameters:
    - model: PyTorch model
    - path: str, the path to save the model.
    """
    file = os.path.join(path, name)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), file)


def get_and_split_files(directory, validation_size):
    """
    Get the files in the directory and split them into training and validation sets.

    Parameters:
    - directory: String, the directory containing the files.
    - validation_size: Float, the size of the validation set.

    Returns:
    - files_train: List of strings, the training files.
    - files_val: List of strings, the validation files.
    """

    files = glob.glob(os.path.join(directory, '*.h5'))
    files_train, files_val = train_test_split(files, test_size=validation_size)

    return files_train, files_val


def extract_batch(X, indices, device):
    """
    Extract a batch from the input tensor.

    Parameters:
    - X: Tensor, the input tensor.
    - indices: List of integers, the indices of the batch.
    - device: PyTorch device.

    Returns:
    - X_batch: Tensor, the batch tensor.
    """

    x_batch = X[indices]
    X_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
    return X_batch


def setup_optimizer_scheduler(model, initial_learning_rate, gamma, l2_weight):
    """
    Setup the optimizer and scheduler.

    Parameters:
    - model: PyTorch model
    - initial_learning_rate: Float, the initial learning rate.
    - gamma: Float, the gamma value for the scheduler.
    - l2_weight: Float, the weight of the L2 regularization.

    Returns:
    - optimizer: PyTorch optimizer.
    - scheduler: PyTorch scheduler.
    """
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_learning_rate,
                                 weight_decay=l2_weight)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    return optimizer, scheduler


def dimension_correlation(X):
    """
    Compute the average correlation between the dimensions of the input tensor.

    Parameters:
    - X: Tensor, the input tensor.

    Returns:
    - correlation: Float, the average absolute correlation.
    """
    corr_matrix = torch.corrcoef(X.T)
    return torch.mean(torch.abs(torch.triu(corr_matrix, diagonal=1)))


def compute_loss(model, X, infoNCE_loss_function, correlation_weight=0):
    """
    Compute the loss of the model.
    
    Parameters:
    - model: PyTorch model
    - X: Tensor of shape [batch_size, num_features] containing the input features.
    - infoNCE_loss_function: InfoNCELoss, the InfoNCE loss function.

    Returns:
    - total_loss: Float, the total loss.
    - infoNCE_loss: Float, the InfoNCE loss.
    - l1_loss: Float, the L1 loss.
    """
    Y = model(X)
    infoNCE_loss_function.get_masks(X)
    infoNCE_loss = infoNCE_loss_function(Y)
    correlation_loss = correlation_weight * dimension_correlation(Y)
    total_loss = infoNCE_loss + correlation_loss
    return total_loss, infoNCE_loss, correlation_loss


def train_one_batch(model,
                    X,
                    info_nce_loss,
                    correlation_weight,
                    optimizer,
                    gradients,
                    ema_params=None):
    """
    Train the model for one batch.

    Parameters:
    - model: PyTorch model 
    - X: Tensor of shape [batch_size, num_features] containing the input features.
    - info_nce_loss: PyTorch loss function, the InfoNCE loss function.
    - correlation_weight: Float, the weight of the correlation loss.
    - optimizer: PyTorch optimizer.
    - gradients: Dict, the gradients.
    - ema_params: Dict, the EMA parameters.

    Returns:
    - info_nce_loss: Float, the InfoNCE loss.
    - correlation_loss: Float, the correlation loss.
    - gradients: Dict, the gradients
    """
    model.train()
    optimizer.zero_grad()

    total_loss, info_nce_loss, correlation_loss = compute_loss(
        model, X, info_nce_loss, correlation_weight)

    total_loss.backward()

    # Apply Grokfast
    if ema_params is not None:
        gradients = gradfilter_ema(model,
                                   grads=gradients,
                                   alpha=ema_params['alpha'],
                                   lamb=ema_params['lambda'])

    optimizer.step()

    return info_nce_loss.item(), correlation_loss.item(), gradients


def validate(model, X_val, info_nce_loss):
    """
    Validate the model.

    Parameters:
    - model: PyTorch model
    - X_val: Tensor of shape [num_samples, num_features] containing the validation data.
    - info_nce_loss: PyTorch loss function, the InfoNCE loss function.

    Returns:
    - loss: Float, the loss value.
    """
    model.eval()
    with torch.no_grad():
        Y_val = model(X_val)
        info_nce_loss.get_masks(X_val)
        loss = info_nce_loss(Y_val)

    return loss.item()


def perform_domain_embedding(model, original_embeddings, device):
    """
    Embed the original embeddings in a lower-dimensional space.

    Parameters:
    - original_embeddings: np.ndarray

    Returns:
    - np.ndarray: The low-dimensional embeddings.
    """
    X = torch.tensor(original_embeddings, dtype=torch.float32).to(device)
    Y = model(X)
    return Y.detach().cpu().numpy()
