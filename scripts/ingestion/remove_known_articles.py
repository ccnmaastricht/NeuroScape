"""
Remove articles that are already part of the dataset (v1 base or a previously processed delta)
from the merged and cleaned dataframe, so that only new articles are embedded and filtered.
"""

import os
import glob
import h5py
import pandas as pd

from src.utils.parsing import parse_directories, parse_discipline

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']


def normalize_doi(dois):
    """
    Normalize DOIs for matching.

    Parameters:
    - dois: pd.Series

    Returns:
    - dois: pd.Series
    """

    return dois.astype(str).str.strip().str.lower()


def load_known_from_hdf5(directory):
    """
    Load the PubMed IDs and DOIs of all articles stored in a directory of HDF5 shards.

    Parameters:
    - directory: str

    Returns:
    - known: pd.DataFrame with columns 'Pmid' and 'Doi'
    """

    pmids, dois = [], []
    for file_name in glob.glob(os.path.join(directory, '*.h5')):
        with h5py.File(file_name, 'r') as file:
            pmids.extend(file['pmid'][:].tolist())
            dois.extend(doi.decode() if isinstance(doi, bytes) else doi
                        for doi in file['doi'][:])

    return pd.DataFrame({'Pmid': pmids, 'Doi': dois})


if __name__ == '__main__':
    directories = parse_directories()
    discipline = parse_discipline()

    base_directory = os.path.join(BASEPATH,
                                  directories['internal']['update']['base'])
    delta_directory = os.path.join(BASEPATH,
                                   directories['internal']['update']['delta'])
    cleaned_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'], discipline)

    base_df = pd.read_csv(os.path.join(base_directory, 'CSV',
                                       'neuroscience_articles_1999-2023.csv'),
                          usecols=['Pmid', 'Doi'])
    delta_df = load_known_from_hdf5(
        os.path.join(delta_directory, 'HDF5', 'Domain'))
    known_df = pd.concat([base_df, delta_df], ignore_index=True)
    known_pmids = set(known_df['Pmid'].astype(int))
    known_dois = set(normalize_doi(known_df['Doi']))
    print(f'Known articles: {len(base_df)} (base) + {len(delta_df)} (delta).')

    cleaned_file = os.path.join(cleaned_directory, 'articles_merged_cleaned.csv')
    all_file = os.path.join(cleaned_directory, 'articles_merged_cleaned_all.csv')

    # Keep the complete cleaned dataframe so that this step can be rerun
    # (a freshly merged dataframe replaces the stored one)
    if not os.path.exists(all_file) or os.path.getmtime(
            cleaned_file) > os.path.getmtime(all_file):
        os.replace(cleaned_file, all_file)
    df = pd.read_csv(all_file)

    known = df['Pmid'].isin(known_pmids) | normalize_doi(
        df['Doi']).isin(known_dois)
    new_df = df[~known]

    print(f'Articles in cleaned dataframe: {len(df)}')
    print(f'Already known: {known.sum()}')
    print(f'New articles: {len(new_df)}')
    print(new_df['Year'].value_counts().sort_index().to_string())

    new_df.to_csv(cleaned_file, index=False)
