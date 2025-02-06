import os
import json


def load_processed_articles(processed_file):
    """
    Load the processed articles from the processed folder.

    Parameters:
    - processed_file: str

    Returns:
    - processed_articles: set
    """
    processed_articles = set()

    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_articles = set(json.load(f))

    return processed_articles


def save_processed_articles(processed_file, processed_articles):
    """
    Save the processed articles to the processed folder.

    Parameters:
    - processed_file: str
    - processed_articles: set
    """

    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    with open(processed_file, 'w') as f:
        json.dump(list(processed_articles), f)
