import os
import pickle
import tomllib

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
