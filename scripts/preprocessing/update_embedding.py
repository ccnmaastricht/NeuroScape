import os
import glob
import time
import argparse

from tqdm import tqdm

from src.utils.parsing import parse_directories
from src.utils.initial_embedding import load_configurations, unpack_embedding_parameters
from src.utils.load_and_save import save_articles_to_hdf5, load_articles_from_hdf5

from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]


def parse_args():
    """
    Parse the command line arguments.
    
    Returns:
    - args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Re-embedding.')
    parser.add_argument('--model',
                        type=str,
                        default='voyage-lite-02-instruct',
                        help='The embedding model.')

    return parser.parse_args()


def parse_model():
    """
    Get the model from the command line arguments.    

    Returns:
    - model: str
    """

    model = parse_args().model

    return model


if __name__ == "__main__":

    directories = parse_directories()

    embedding_parameters = load_configurations()
    _, sleep_time, batch_size, items_per_shard = unpack_embedding_parameters(
        embedding_parameters)

    model = parse_model()
    embedding_model = VoyageAIEmbeddings(model=model,
                                         batch_size=batch_size,
                                         voyage_api_key=VOYAGE_API_KEY)

    articles_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['voyage'])

    article_shards = glob.glob(os.path.join(articles_directory, '*.h5'))

    temp_directory = articles_directory + '_temp'
    os.makedirs(temp_directory, exist_ok=True)

    num_shards = len(article_shards)

    print('Updating the embeddings...')
    for i, shard in tqdm(enumerate(article_shards), total=num_shards):

        articles = load_articles_from_hdf5(shard, disable_tqdm=True)
        abstracts = [article.abstract for article in articles]
        embedded_abstracts = embedding_model.embed_documents(abstracts)

        for article, embedded_abstract in zip(articles, embedded_abstracts):
            article.embedding = embedded_abstract

        output_file = os.path.join(temp_directory, f'shard_{i:04d}.h5')
        save_articles_to_hdf5(articles, output_file, disable_tqdm=True)
        time.sleep(sleep_time)

    os.rename(articles_directory, articles_directory + 'Original')
    os.rename(temp_directory, articles_directory)
