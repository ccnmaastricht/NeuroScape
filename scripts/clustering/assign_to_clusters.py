"""
Assign articles added after clustering (NeuroScape 2.0) to the existing clusters.

Each new article receives the cluster with the largest similarity-weighted vote among its k nearest
clustered (v1) articles in domain embedding space. Only v1 articles act as neighbours, so the v1
clustering stays the fixed reference.

With --validate, articles from the validation year are held out instead, assigned using the remaining
v1 articles, and compared with their Leiden labels.
"""

import os
import argparse
import faiss
import numpy as np
import pandas as pd
from glob import glob

from src.utils.clustering import load_configurations
from src.utils.parsing import parse_directories
from src.utils.load_and_save import load_embedding_shards

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']


def parse_args():
    """
    Parse the command line arguments.

    Returns:
    - args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Assign new articles to existing clusters.')
    parser.add_argument('--validate',
                        action='store_true',
                        help='Validate on held-out v1 articles instead.')

    return parser.parse_args()


def normalize(embeddings):
    """
    L2-normalize embeddings.

    Parameters:
    - embeddings: np.array

    Returns:
    - embeddings: np.array (float32)
    """

    embeddings = embeddings.astype(np.float32)

    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def knn_vote(reference_embeddings, reference_labels, query_embeddings,
             num_neighbors):
    """
    Assign query embeddings by a similarity-weighted vote of their nearest reference embeddings.

    Parameters:
    - reference_embeddings: np.array, L2-normalized
    - reference_labels: np.array of int
    - query_embeddings: np.array, L2-normalized
    - num_neighbors: int

    Returns:
    - labels: np.array of int, the assigned clusters
    - shares: np.array of float, weighted vote share of the assigned cluster
    - margins: np.array of float, share of the assigned minus the runner-up cluster
    """

    index = faiss.IndexFlatIP(reference_embeddings.shape[1])
    index.add(reference_embeddings)
    similarities, neighbors = index.search(query_embeddings, num_neighbors)
    similarities = np.clip(similarities, 0, None)

    num_clusters = reference_labels.max() + 1
    votes = np.zeros((len(query_embeddings), num_clusters))
    np.add.at(votes, (np.arange(len(query_embeddings))[:, None],
                      reference_labels[neighbors]), similarities)
    votes /= votes.sum(axis=1, keepdims=True)

    ranked = np.sort(votes, axis=1)
    labels = votes.argmax(axis=1)
    shares = ranked[:, -1]
    margins = ranked[:, -1] - ranked[:, -2]

    return labels, shares, margins


if __name__ == '__main__':
    args = parse_args()
    configurations = load_configurations()['assignment']
    num_neighbors = configurations['num_neighbors']
    directories = parse_directories()

    base_directory = os.path.join(BASEPATH,
                                  directories['internal']['update']['base'])
    delta_directory = os.path.join(BASEPATH,
                                   directories['internal']['update']['delta'])
    new_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['neuro'])
    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'],
        'Neuroscience')

    print('Loading clustered (v1) articles...')
    base_embeddings, base_pmids = load_embedding_shards(
        glob(os.path.join(base_directory, 'HDF5', 'DomainEmbeddings', '*.h5')))
    base_df = pd.read_csv(os.path.join(base_directory, 'CSV',
                                       'neuroscience_articles_1999-2023.csv'),
                          usecols=['Pmid', 'Year', 'Cluster ID'])
    base_df = base_df.set_index('Pmid').loc[base_pmids]
    base_embeddings = normalize(base_embeddings)
    base_labels = base_df['Cluster ID'].values

    if args.validate:
        validation_year = configurations['validation_year']
        held_out = base_df['Year'].values == validation_year
        labels, shares, margins = knn_vote(base_embeddings[~held_out],
                                           base_labels[~held_out],
                                           base_embeddings[held_out],
                                           num_neighbors)
        agreement = labels == base_labels[held_out]

        print(f'Validation on {held_out.sum()} articles from {validation_year} '
              f'(k = {num_neighbors}).')
        print(f'Agreement with Leiden labels: {agreement.mean():.3f}')
        for lower, upper in [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
            selection = (shares >= lower) & (shares < upper)
            print(f'  vote share {lower:.1f}-{min(upper, 1):.1f}: '
                  f'{selection.mean():.1%} of articles, agreement '
                  f'{agreement[selection].mean():.3f}')

    else:
        print('Loading new articles...')
        new_files = glob(os.path.join(delta_directory, 'HDF5', 'Domain',
                                      '*.h5')) + glob(
                                          os.path.join(new_directory, '*.h5'))
        new_embeddings, new_pmids = load_embedding_shards(new_files)
        new_embeddings = normalize(new_embeddings)

        print(f'Assigning {len(new_pmids)} articles (k = {num_neighbors})...')
        labels, shares, margins = knn_vote(base_embeddings, base_labels,
                                           new_embeddings, num_neighbors)

        assignment_df = pd.DataFrame({
            'Pmid': new_pmids,
            'Cluster ID': labels,
            'Assignment Share': shares,
            'Assignment Margin': margins
        })
        output_file = os.path.join(csv_directory, 'articles_assigned.csv')
        os.makedirs(csv_directory, exist_ok=True)
        assignment_df.to_csv(output_file, index=False)

        print(f'Saved assignments to {output_file}.')
        print(f'Median vote share: {np.median(shares):.3f}; '
              f'share < 0.5: {(shares < 0.5).mean():.1%}')
