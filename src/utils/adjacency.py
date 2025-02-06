import os
import tomllib
from Bio import Entrez
from habanero import Crossref


def load_configurations():
    """
    Load the configuration from the config file.
    
    Returns:
    - configurations: dict
    """
    with open('config/ingestion/adjacency.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def find_doi(pubmed_id, articles):
    """
    Find the DOI of the article with the given PubMed ID.

    Parameters:
    - pubmed_id: int
        PubMed ID of the article.
    - articles: list
        List of Article objects.

    Returns:
    - doi: str
        DOI of the article.
    """

    # Finding the DOI of the article with the given PubMed ID
    doi = None
    for article in articles:
        if article.pmid == pubmed_id:
            doi = article.doi
            break

    return doi


def find_pubmed_id(doi, articles):
    """
    Find the PubMed ID of the article with the given DOI.

    Parameters:
    - doi: str
        DOI of the article.
    - articles: list
        List of Article objects.

    Returns:
    - pubmed_id: int
        PubMed ID of the article.
    """

    pubmed_id = None
    for article in articles:
        if article.doi == doi:
            pubmed_id = article.pmid
            break

    return pubmed_id


def fetch_in_links(pubmed_id):
    """
    Fetch the PubMed IDs of articles that the given article cites.

    Parameters:
    - pubmed_id: int
        PubMed ID of the article.

    Returns:
    - pubmed_ids: list
        List of PubMed IDs.
    """

    # Using elink to fetch the PubMed IDs of articles that cite the given article
    link_result = Entrez.elink(id=f'{pubmed_id}',
                               linkname="pubmed_pubmed_citedin")
    record = Entrez.read(link_result)
    link_result.close()

    # Extracting PubMed IDs from the links
    pubmed_ids = []
    if record[0]["LinkSetDb"]:
        links = record[0]["LinkSetDb"][0]["Link"]
        pubmed_ids = [int(link["Id"]) for link in links]

    return pubmed_ids


def fetch_out_links(doi):
    """
    Fetch the DOIs of articles that the given article cites.

    Parameters:
    - doi: str
        DOI of the article.

    Returns:
    - dois: list
        List of DOIs.
    """
    cr = Crossref()
    # Fetch article metadata by DOI
    metadata = cr.works(ids=doi)
    references = metadata['message'].get('reference', [])

    # Extracting DOIs from references, if available
    ref_dois = [ref.get('DOI') for ref in references if 'DOI' in ref]
    return ref_dois


def get_intersection(a, b):
    """ 
    Get the intersection of two lists.
    
    Parameters:
    - a: list
        First list.
    - b: list
        Second list.
        
    Returns:
    - intersection: list
        Intersection of the two lists.
    """

    return list(set(a).intersection(set(b)))
