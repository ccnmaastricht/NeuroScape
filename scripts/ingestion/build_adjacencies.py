import os
import numpy as np
from time import sleep
from tqdm import tqdm
from Bio import Entrez
from glob import glob

from src.utils.adjacency import *
from src.utils.parsing import parse_directories
from src.utils.checkpoints import load_processed_articles, save_processed_articles
from src.utils.load_and_save import save_articles_to_hdf5, load_articles_from_hdf5, determine_output_filename

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']
EMAIL = os.environ['EMAIL']


def fetch_links(articles, all_pubmed_ids, config, items_per_shard,
                article_directory, utils_folder):
    """
    Fetch the in-links and out-links for the given articles.

    Parameters:
    - articles: list
        List of Article objects.
    - all_pubmed_ids: list
        List of all PubMed IDs.
    - config: dict
        Configuration for fetching the links.
    - items_per_shard: int 
        Number of items per shard.
    - article_directory: str
        Directory for the articles.
    - utils_folder: str
        Directory for the utility files.

    Returns:
    - articles: list
        List of Article objects with in-links and out-links added.
    """

    num_attempts = config['num_attempts']
    sleep_time = config['sleep_time']

    all_dois = [article.doi for article in articles]
    doi_to_pubmed_id = {article.doi: article.pmid for article in articles}

    output_directory = os.path.join(article_directory, 'backup')
    os.makedirs(output_directory, exist_ok=True)
    output_file, shard_id = determine_output_filename(output_directory, 'h5')

    num_items = 0
    processed_file = os.path.join(utils_folder, 'linked_articles.json')
    processed_articles = load_processed_articles(processed_file)
    # Fetching the in-links and out-links
    shard = []
    for article in tqdm(articles, total=len(articles)):

        pubmed_id = article.pmid
        doi = article.doi

        if pubmed_id in processed_articles:
            continue

        for _ in range(num_attempts):
            try:
                in_link_candidates = fetch_in_links(pubmed_id)
                break
            except:
                sleep(sleep_time)

        for _ in range(num_attempts):
            try:
                out_link_candidates = fetch_out_links(doi)
                break
            except:
                sleep(sleep_time)

        in_links = get_intersection(in_link_candidates, all_pubmed_ids)
        out_dois = get_intersection(out_link_candidates, all_dois)
        out_links = [doi_to_pubmed_id[doi] for doi in out_dois]

        article.in_links = in_links
        article.out_links = out_links
        shard.append(article)
        processed_articles.add(pubmed_id)
        num_items += 1

        if (num_items == items_per_shard):
            save_articles_to_hdf5(shard, output_file, disable_tqdm=True)
            save_processed_articles(processed_file, processed_articles)
            shard = []
            num_items = 0
            shard_id += 1
            output_file = os.path.join(output_directory,
                                       f'shard_{shard_id:04d}.h5')

    return articles


def update_links(articles, all_pubmed_ids):
    """ 
    Update the in-links and out-links for the given articles.
    
    Parameters:
    - articles: list
        List of Article objects.
    - all_pubmed_ids: list
        List of all PubMed IDs.

    Returns:
    - articles: list
        List of Article objects with updated in-links and out-links.        
    """
    # Create a dictionary mapping PubMed IDs to indices
    id_to_index = {
        pubmed_id: index
        for index, pubmed_id in enumerate(all_pubmed_ids)
    }

    for article in tqdm(articles, total=len(articles)):
        for out_link in article.out_links:
            article_index = id_to_index[out_link]
            articles[article_index].in_links = list(
                set(articles[article_index].in_links) | {article.pmid})

        for in_link in article.in_links:
            article_index = id_to_index[in_link]
            articles[article_index].out_links = list(
                set(articles[article_index].out_links) | {article.pmid})

    return articles


if __name__ == '__main__':
    configurations = load_configurations()
    Entrez.email = EMAIL
    fetch_config = configurations['pubmed_requests']
    items_per_shard = configurations['storage']['items_per_shard']

    directories = parse_directories()
    article_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['voyage'])

    checkpoints_folder = os.path.join(BASEPATH,
                                      directories['internal']['checkpoints'])

    # Loading the articles
    print('Loading articles...')
    file_names = glob(os.path.join(article_directory, '*.h5'))
    all_articles = []

    file_pmid_dict = {}
    for file_name in tqdm(file_names):
        articles = load_articles_from_hdf5(file_name, disable_tqdm=True)
        all_articles.extend(articles)
        file_pmid_dict[file_name] = [article.pmid for article in articles]

    all_pubmed_ids = [article.pmid for article in all_articles]

    # Fetching the in-links and out-links
    print('Fetching links...')
    all_articles = fetch_links(all_articles, all_pubmed_ids, fetch_config,
                               items_per_shard, article_directory,
                               checkpoints_folder)
    print('Updating links...')
    all_articles = update_links(all_articles, all_pubmed_ids)

    # Saving the articles with in-links and out-links
    print('Saving articles...')

    for file_name in tqdm(file_names):
        articles = [
            article for article in all_articles
            if article.pmid in file_pmid_dict[file_name]
        ]

        # replace the old file with the new one
        save_articles_to_hdf5(articles, file_name, disable_tqdm=True)
