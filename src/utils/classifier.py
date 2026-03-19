import os
import torch
import pickle
import tomllib
import numpy as np

from sklearn.metrics import accuracy_score

from src.classes.data_types import EmbeddingsWithLabels


def load_configurations():
    """
    Load the configuration from the config file.
    
    Returns:
    - configurations: dict
    """
    with open('config/preprocessing/classifier.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


# Utility functions for preparing the data for training the classifier


def get_unique_disciplines_and_count(dataframe):
    """
    Get the unique disciplines and the number of classes.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the disciplines.

    Returns:
    list: A list of unique disciplines.
    int: The number of classes.
    """
    disciplines = [
        get_disciplines(dataframe, pmid) for pmid in dataframe['Pmid'].values
    ]
    unique_disciplines = list(
        set([item for sublist in disciplines for item in sublist]))
    unique_disciplines = sorted(unique_disciplines)
    num_classes = len(unique_disciplines)
    return unique_disciplines, num_classes


def load_shard(shard_path):
    """ 
    Load a shard from a pickle file.
    
    Parameters:
    shard_path (str): The path to the shard.
    
    Returns:
    EmbeddingsWithLabels: The embeddings and labels of the shard.
    """
    with open(shard_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_shard(data, shard_path):
    """
    Save a shard to a pickle file.

    Parameters:
    data (EmbeddingsWithLabels): The embeddings and labels to be saved.
    shard_path (str): The path to the shard.
    """
    with open(shard_path, 'wb') as f:
        pickle.dump(data, f)


def extract_data(abstracts, n_hot_vector, num_hot, condition):
    """ 
    Extract data from a list of embeddings and labels given a condition.
    
    Parameters:
    abstracts (EmbeddingsWithLabels): The embeddings and labels.
    n_hot_vector (list): The n-hot vector of the embeddings.
    num_hot (list): The number of disciplines of the embeddings.
    condition (function): The condition to be satisfied.
    
    Returns:
    list: A list of pubmed ids.
    list: A list of embeddings.
    list: A list of labels.
    """
    pmids = [abstracts.pmids[i] for i, n in enumerate(num_hot) if condition(n)]
    embeddings = [
        abstracts.embeddings[i] for i, n in enumerate(num_hot) if condition(n)
    ]
    labels = [n_hot_vector[i] for i, n in enumerate(num_hot) if condition(n)]
    return pmids, embeddings, labels


def save_and_create_dataset(data, id, directory, set_type='Train'):
    """
    Save the current data to a shard and create a new dataset.

    Parameters:
    data (EmbeddingsWithLabels): The embeddings and labels to be saved.
    id (int): The id of the shard.
    directory (str): The directory where the shard will be saved.
    set_type (str): The type of the dataset.

    Returns:
    EmbeddingsWithLabels: The new dataset.
    int: The new id of the shard.
    """
    file_name = f'shard_{id:04d}.pkl'
    save_shard(data, os.path.join(directory, set_type, file_name))
    data = EmbeddingsWithLabels(pmids=[], embeddings=[], labels=[])
    return data, id + 1


def generate_n_hot_vector(shard_disciplines, unique_disciplines):
    """
    Generate the n-hot vector of a list of disciplines.

    Parameters:
    shard_disciplines (list): A list of disciplines.
    unique_disciplines (list): A list of unique disciplines.

    Returns:
    list: A list of n-hot vectors.
    list: A list of the number of positive instances.
    """
    n_hot_vector = [[
        1 if discipline in disciplines else 0
        for discipline in unique_disciplines
    ] for disciplines in shard_disciplines]
    num_hot = [sum(subset) for subset in n_hot_vector]
    return n_hot_vector, num_hot


def get_disciplines(dataframe, pmids):
    """
    Get the disciplines of a paper given its pubmed id(s).

    Parameters:
    - dataframe: pd.DataFrame
    - pmids: int or list

    Returns:
    - list: A list of disciplines.
    """
    if isinstance(pmids, list):
        return [
            dataframe['Disciplines'][dataframe['Pmid'] ==
                                     pmid].values[0].split(' / ')
            for pmid in pmids
        ]
    else:
        return dataframe['Disciplines'][dataframe['Pmid'] ==
                                        pmids].values[0].split(' / ')


# Utility functions for training the classifier


def train_one_epoch(model, X, Y, loss_function, optimizer):
    """
    Train the model for one epoch.

    Parameters:
    - model: PyTorch model 
    - X: Tensor of shape [batch_size, num_features] containing the input features.
    - Y: Tensor of shape [batch_size, num_classes] containing the true class labels.
    - loss_function: PyTorch loss function.
    - optimizer: PyTorch optimizer.

    Returns:
    - loss: Float, the loss value.
    """
    model.train()
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = loss_function(Y_pred, Y)
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_multiclass_accuracy(Y_pred, Y_true):
    """
    Compute the multiclass accuracy.

    Parameters:
    - Y_pred: Numpy array of shape [batch_size, num_classes] containing the model's predictions.
    - Y_true: Numpy array of shape [batch_size, num_classes] containing the true class labels.

    Returns:
    - accuracy: Float, the multiclass accuracy.
    """
    Y_pred_classes = (Y_pred > 0.5).astype(int)
    Y_true_classes = Y_true.astype(int)
    accuracy = accuracy_score(Y_true_classes, Y_pred_classes)
    return accuracy


def compute_expected_accuracy(Y):
    """
    Compute the expected accuracy.

    Parameters:
    - Y: Numpy array of shape [num_samples, num_classes] containing the class labels.

    Returns:
    - expected_accuracy: Float, the expected accuracy.
    """
    items_per_class = np.sum(Y, axis=0)
    total_items = np.sum(items_per_class)
    expected_accuracy = np.sum((items_per_class / total_items)**2)
    return expected_accuracy


def compute_kappa(accuracy, expected_accuracy):
    """
    Compute the Cohen's Kappa.

    Parameters:
    - accuracy: Float, the multiclass accuracy.
    - expected_accuracy: Float, the expected accuracy.

    Returns:
    - kappa: Float, the Cohen's Kappa.
    """
    kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)
    return kappa


def validate(model, X, Y, loss_function):
    """
    Validate the model on the validation set.

    Parameters:
    - model: PyTorch model
    - X: Tensor of shape [batch_size, num_features] containing the input features.
    - Y: Tensor of shape [batch_size, num_classes] containing the true class labels.
    - loss_function: PyTorch loss function.

    Returns:
    - loss: Float, the loss value.
    - accuracy: Float, the multiclass accuracy.
    """
    model.eval()
    Y_pred = model(X)
    loss = loss_function(Y_pred, Y)
    accuracy = compute_multiclass_accuracy(Y_pred.cpu().detach().numpy(),
                                           Y.cpu().detach().numpy())
    return loss.item(), accuracy


def data_loader(files):
    """
    Load the data from the files.

    Parameters:
    - files: List of str, the paths to the files.

    Returns:
    - X: Numpy array of shape [num_samples, num_features] containing the input features.
    - Y: Numpy array of shape [num_samples, num_classes] containing the class labels.
    """
    X = []
    Y = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        X.extend(data.embeddings)
        Y.extend(data.labels)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def to_device(X, Y, device):
    """
    Move the data to the device.

    Parameters:
    - X: Numpy array of shape [num_samples, num_features] containing the input features.
    - Y: Numpy array of shape [num_samples, num_classes] containing the class labels.
    - device: PyTorch device.

    Returns:
    - X: Tensor of shape [num_samples, num_features] containing the input features.
    - Y: Tensor of shape [num_samples, num_classes] containing the class labels.
    """
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    return X, Y


def drop_class(model, X, Y, device, confidence_cutoff):
    X, _ = to_device(X, Y, device)

    #se X for vazio, retorna Y sem alterações
    if not isinstance(X, torch.Tensor) or X.shape[1] == 0:
        print("drop_class: entrada inválida detectada, arquivo ignorado")
        return Y

    Y_pred = model(X)
    class_probs = Y_pred.cpu().detach().numpy()
    class_probs *= Y
    confidence = class_probs / np.max(class_probs, axis=1).reshape(-1, 1)
    drop_indices = confidence < confidence_cutoff

    Y[drop_indices] = 0
    return Y


def save_model(model, path):
    """
    Save the model to a file.

    Parameters:
    - model: PyTorch model
    - path: str, the path to save the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
