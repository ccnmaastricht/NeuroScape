import os
import tomllib
import tqdm
import pandas as pd

from crossref.restful import Works


def load_configurations():
    """
    Load the configuration from the config file.
    
    Returns:
    - configurations: dict
    """
    with open('config/ingestion/cleaning.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def concatenate_files(files):
    """
    Concatenate the files using pandas.concat.
    
    Parameters:
    - files: list
    
    Returns:
    - df: pandas.DataFrame
    """
    dfs = []
    for file in tqdm.tqdm(files):
        df = pd.read_csv(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df


def sort_dataframe(df):
    """
    Sort the dataframe by the 'Year' and then by 'Journal' within each year.

    Parameters:
    - df: pandas.DataFrame

    Returns:
    - df: pandas.DataFrame
    """

    df = df.sort_values(by=['Year', 'Journal', 'Type'])
    return df


def remove_duplicate_doi(df):
    """
    Remove duplicate dois from the dataframe.
    
    Parameters:
    - df: pandas.DataFrame
    
    Returns:
    - df: pandas.DataFrame
    """
    works = Works()

    duplicate_list = df[df.duplicated(['Doi'],
                                      keep=False)]['Pmid'].values.tolist()

    for pmid in tqdm.tqdm(duplicate_list):
        doi = df.query('Pmid == @pmid')['Doi'].values[0]
        original_title = df.query('Pmid == @pmid')['Title'].values[0]
        if works.doi(doi) is None:
            continue
        retrieved_title = works.doi(doi)['title']
        if retrieved_title is None or len(
                retrieved_title) == 0 or original_title == retrieved_title:
            continue

        # remove entry from df
        drop_index = df.index[df['Pmid'] == pmid].tolist()
        df = df.drop(drop_index)

    # remove duplicate dois
    df.drop_duplicates(subset=['Doi'], inplace=True)

    return df


def count_words(text):
    """
    Count the number of words in a text.

    Parameters:
    - text: str

    Returns:
    - count: int
    """

    count = len(text.split())
    return count


def get_abstract_drop_index(df, lower, upper):
    """
    Get the index of the rows with abstracts that are too short or too long.

    Parameters:
    - df: pandas.DataFrame
    - lower: int
    - upper: int

    Returns:
    - drop_index: list
    """
    drop_index = df.index[df['Abstract'].apply(count_words) < lower].tolist()
    drop_index += df.index[df['Abstract'].apply(count_words) > upper].tolist()
    return drop_index


def clean_dataframe(df, cutoffs):
    """
    Clean the dataframe by removing rows with missing values, duplicates, etc.

    Parameters:
    - df: pandas.DataFrame

    Returns:
    - df: pandas.DataFrame
    """

    lower_word_limit, upper_word_limit, year_cutoff = cutoffs

    # remove all rows with pmid, doi, abstract, or year is none or nan or inf, and abstracts that are too short or too long
    drop_index = df.index[df['Pmid'].isnull() | df['Doi'].isnull()
                          | df['Abstract'].isnull()
                          | df['Year'].isnull()].tolist()
    drop_index.extend(
        get_abstract_drop_index(df, lower_word_limit, upper_word_limit))
    drop_index.extend(df.index[df['Year'] > year_cutoff].tolist())
    df = df.drop(drop_index)

    # remove rows with duplicate pmids
    df.drop_duplicates(subset=['Pmid'], inplace=True)

    # remove rows with duplicate dois
    df = remove_duplicate_doi(df)

    return df
