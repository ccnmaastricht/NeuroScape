import time
import tomllib
import numpy as np


def load_configurations():
    """
    Load the configuration from the config file.
    
    Returns:
    - configurations: dict
    """
    with open('config/analysis/semantic.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def validate_dictionary(dictionary, required_fields):
    """
    Validate the dictionary extracted from the LLM output.

    Parameters:
    - dictionary: dict, the LLM output dictionary.
    - required_fields: list, the required fields in the dictionary.


    Returns:
    - dictionary: dict, the extracted dictionary.
    """

    try:
        # Verify that the required fields are present
        if not all(key in dictionary for key in required_fields):
            raise ValueError("Missing required fields in the LLM output")

        return dictionary

    except Exception:
        return None


def safe_dictionary_extraction(required_fields, chain_input, chain, retries,
                               delay):
    """
    Safely extract the dictionary from the LLM output.

    Parameters:
    - cluster_title: String, the cluster
    - abstracts_text: String, the abstracts text.
    - chain: LLMChain, the LLM chain to invoke.
    - retries: int, the number of retries allowed.
    - delay: int, the delay between retries.

    Returns:
    - dictionary: dict, the extracted dictionary.
    """
    attempt = 0
    while attempt < retries:
        dictionary = chain.invoke(chain_input)
        cluster_dimensions = validate_dictionary(dictionary, required_fields)
        time.sleep(delay)
        if cluster_dimensions:
            return cluster_dimensions
        else:
            attempt += 1

    return None


def get_abstract_strings(centroid, embeddings, abstracts, number_of_abstracts):
    """
    Get the abstract strings that are most similar to the centroid.

    Parameters:
    - centroid: np.array, the centroid of the cluster.
    - embeddings: np.array, the embeddings of the cluster.
    - abstracts: np.array, the abstracts of the cluster.

    Returns:
    - abstracts: str, the abstract strings.
    """

    similarity = centroid.dot(embeddings.T)
    top_indices = np.argsort(similarity)[::-1][:number_of_abstracts]

    return '\n\n'.join(abstracts[top_indices])
