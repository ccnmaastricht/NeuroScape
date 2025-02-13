import re
import time
import tomllib
import numpy as np

from pdfminer.high_level import extract_text


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


def strip_document(document):
    """
    Strip the document of any unwanted sections.

    Parameters:
    - document: str, the document to strip.

    Returns:
    - document: str, the stripped document.
    """

    unnecessary_sections = [
        'references', 'acknowledgements', 'author contributions', 'funding',
        'funding sources', 'conflict of interest',
        'conflict of interest statement'
    ]

    # Convert document to lowercase for case-insensitive matching
    document_lower = document.lower()

    # Use regular expressions to strip the document starting from 'introduction' or 'main'
    introduction_match = re.search(r'\bintroduction\b', document_lower,
                                   re.IGNORECASE)
    main_match = re.search(r'\bmain\b', document_lower, re.IGNORECASE)

    if introduction_match:
        document = document[introduction_match.end():]
    elif main_match:
        document = document[main_match.end():]

    # Use regular expressions to remove unnecessary sections
    for section in unnecessary_sections:
        pattern = re.compile(r'\b' + re.escape(section) + r'\b', re.IGNORECASE)
        match = pattern.search(document)
        if match:
            document = document[:match.start()]

    return document


def get_review_text(files):
    """
    Get the review text from a list of pdf files.

    Parameters:
    - files: list, the list of pdf files to extract text from.

    Returns:
    - review_text: str, the review text.
    """

    review_text = ''
    for file in files:
        document = extract_text(file)
        document = strip_document(document)
        review_text = f"""{review_text}\n\n{document}"""

    return review_text
