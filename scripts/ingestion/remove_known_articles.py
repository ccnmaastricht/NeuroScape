"""
Remove articles that are already part of the dataset (v1 base or a previously processed delta)
from the merged and cleaned dataframe, so that only new articles are embedded and filtered.
"""

import os
import pandas as pd

from src.utils.parsing import parse_directories, parse_discipline
from src.utils.update import normalize_doi, load_known_from_hdf5

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']


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

    # Filtering is idempotent; rerun merge_and_clean.py to start from the complete dataframe
    cleaned_file = os.path.join(cleaned_directory, 'articles_merged_cleaned.csv')
    df = pd.read_csv(cleaned_file)

    known = df['Pmid'].isin(known_pmids) | normalize_doi(
        df['Doi']).isin(known_dois)
    new_df = df[~known]

    print(f'Articles in cleaned dataframe: {len(df)}')
    print(f'Already known: {known.sum()}')
    print(f'New articles: {len(new_df)}')
    print(new_df['Year'].value_counts().sort_index().to_string())

    new_df.to_csv(cleaned_file, index=False)
