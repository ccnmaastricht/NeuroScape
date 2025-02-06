import pickle
import tomllib


def load_configurations():
    """
    Load the configurations from the config files.
    
    Returns:
    - data_directories: dict
    - embedding: dict
    """

    with open(f'config/ingestion/initial_embedding.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def unpack_embedding_parameters(config):
    """
    Unpack the embedding parameters from the embedding dict.
    
    Parameters:
    - config: dict
    
    Returns:
    - items_per_shard: int
    """

    model = config['model']
    sleep_time = config['api_calls']['sleep_time']
    batch_size = config['api_calls']['batch_size']
    items_per_shard = config['items_per_shard']

    return model, sleep_time, batch_size, items_per_shard


def save_embeddings(embeddings, output_file):
    """
    Save the embeddings to the output file.

    Parameters:
    - embeddings: Embeddings
    - output_file: str
    """
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
