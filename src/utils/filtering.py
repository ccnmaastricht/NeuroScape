import os
import tomllib
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from datetime import datetime

from src.classes.data_types import Article
from src.classes.discipline_classifier import DisciplineClassifier

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month


def load_configurations():
    """
    Load the configuration from the config file.

    Returns:
    - configurations: dict
    """
    with open('config/preprocessing/classifier.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def update_keep_index(keep_index, pubmed_ids, df):
    """
    Update the keep index with the new pubmed ids.
    
    Parameters:
    - keep_index: list
    - pubmed_ids: list
    - df: pd.DataFrame

    Returns:
    - new_keep_index: list
    """

    new_keep_index = df.index[df['Pmid'].isin(pubmed_ids)].tolist()
    keep_index.extend(new_keep_index)

    return keep_index


def update_article_data(data, new_data):
    """
    Update the article data with the new data.
    """

    data.extend(new_data)

    return data


def is_neuroscience(discipline):
    """
    Check if an article's discipline includes Neuroscience according
    to the Journal it was published in.
    
    Parameters:
    - disciplines: list
    
    Returns:
    - is_neuroscience: list
    """

    return 'Neuroscience' in discipline


def is_unique(discipline):
    """
    Check if an article is mono-disciplinary according to the Journal
    it was published in.
    
    Parameters:
    - disciplines: list
    
    Returns:
    - is_unique: list
    """

    return len(discipline) == 1


def remove(disciplines, confidence, cutoff):
    """
    Mark articles for removal based on the disciplines of the Journal
    it was published in and the probability of being neuroscience 
    according to the filter network.

    Parameters:
    - disciplines: list
    - confidence: np.array of float
    - cutoff: float

    Returns:
    - remove_index: np.array of bool
    """

    candidates = np.array([
        not is_neuroscience(discipline) or not is_unique(discipline)
        for discipline in disciplines
    ])

    return np.logical_and(candidates, confidence < cutoff)


def compute_article_age(df):
    """
    Computes the age of single article in years.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the articles.
        
    Returns
    -------
    age_in_years : np.array
    """
    month = pd.to_datetime(df['Month'], format='%b').dt.month
    year = df['Year'].astype(int)

    age_in_months = (CURRENT_YEAR - year) * 12 + (CURRENT_MONTH -
                                                  month.fillna(1))
    age_in_years = age_in_months / 12

    return age_in_years.values[0]


def compute_citation_rate(citation_count, article_age):
    """
    Compute the citation rate of an article.
    """

    return citation_count / article_age


def fill_article(pmid, dataframe, embedding):
    """
    Fill the article data class with the relevant information.
    """

    article = dataframe.loc[dataframe['Pmid'] == pmid]
    age = compute_article_age(article)
    citation_rate = compute_citation_rate(article['Citations'].values[0], age)

    return Article(pmid, article['Doi'].values[0], article['Title'].values[0],
                   article['Type'].values[0], article['Journal'].values[0],
                   int(article['Year'].values[0]), age,
                   article['Citations'].values[0], citation_rate,
                   article['Abstract'].values[0], embedding, [], [])


def load_model(configurations, model_file):
    """
    Load the filter network model.

    Parameters:
    - configurations: dict
    - model_file: str

    Returns:
    - model: DisciplineClassifier
    """

    device = configurations['device']
    layer_sizes = configurations['layer_sizes']
    num_classes = configurations['num_classes']

    model = DisciplineClassifier(layer_sizes, num_classes)
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.eval()

    return model
