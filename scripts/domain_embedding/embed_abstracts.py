import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

from src.utils.parsing import parse_directories
from src.classes.sparse_embedding_network import SparseEmbeddingNetwork
from src.utils.domain_embedding import load_configurations, perform_domain_embedding
from src.utils.load_and_save import load_articles_from_hdf5, save_articles_to_hdf5

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']


def load_embedding_network(input_dimension, hidden_dimensions,
                           output_dimension, device, model_file):
    """
    Load the embedding network.

    Parameters:
    - input_dimension: int, the input dimension.
    - hidden_dimensions: list, the hidden dimensions.
    - output_dimension: int, the output dimension.
    - dropout: float, the dropout rate.
    - device: str, the device to use.
    - model_file: str, the path to the model file.

    Returns:
    - embedding_network: SparseEmbeddingNetwork, the embedding network.
    """

    embedding_network = SparseEmbeddingNetwork(
        input_dimension=input_dimension,
        hidden_dimensions=hidden_dimensions,
        output_dimension=output_dimension).to(device)

    embedding_network.load_state_dict(torch.load(model_file))
    embedding_network.eval()
    return embedding_network


if __name__ == '__main__':
    # Load Configurations
    configurations = load_configurations()

    device = configurations['model']['device']
    input_dimension = configurations['model']['input_dimension']
    hidden_dimensions = configurations['model']['hidden_dimensions']
    output_dimension = configurations['model']['output_dimension']

    directories = parse_directories()

    original_data_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['voyage'])
    new_data_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['neuro'])
    model_save_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['models'])

    os.makedirs(new_data_directory, exist_ok=True)

    model_file = os.path.join(model_save_directory,
                              'domain_embedding_model_best.pth')

    embedding_network = load_embedding_network(input_dimension,
                                               hidden_dimensions,
                                               output_dimension, device,
                                               model_file)

    article_files = glob(os.path.join(original_data_directory, '*.h5'))
    print(f'Found {len(article_files)} article files.')
    for article_file in tqdm(article_files):
        articles = load_articles_from_hdf5(article_file, disable_tqdm=True)
        for article in articles:
            embedding = np.array(article.embedding).reshape(1, -1)

            article.embedding = perform_domain_embedding(
                embedding_network, embedding, device).squeeze(axis=0).tolist()

        new_filename = os.path.join(new_data_directory,
                                    os.path.basename(article_file))
        save_articles_to_hdf5(articles, new_filename, disable_tqdm=True)

    print('Done.')
