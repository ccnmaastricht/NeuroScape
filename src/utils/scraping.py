import tomllib
import pandas as pd

from Bio import Entrez

from src.utils.parsing import parse_args


def load_configurations():
    """
    Load the configurations from the config files.
    
    Returns:
    - data_directories: dict
    - scraping: dict
    """

    with open('config/ingestion/scraping.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def unpack_scraping_parameters(scraping_parameters):
    """
    Unpack the scraping parameters from the scraping dict.
    
    Parameters:
    - scraping_parameters: dict
    
    Returns:
    - email: str
    - prefix: str
    - suffix: str
    - sleep_time: int
    - max_results: int
    - num_attempts: int
    - items_per_shard: int
    """
    prefix = scraping_parameters['setup']['prefix']
    suffix = scraping_parameters['setup']['suffix']

    num_attempts = scraping_parameters['pubmed_requests']['num_attempts']
    sleep_time = scraping_parameters['pubmed_requests']['sleep_time']

    items_per_shard = scraping_parameters['storage']['items_per_shard']

    return prefix, suffix, sleep_time, num_attempts, items_per_shard


def get_id_list(query, email, max_results=5000):
    """
    Gets a list of PubMed IDs for a given query.
    
    Parameters
    ----------
    query : str
        Query to search PubMed with.

    Returns
    -------
    id_list : list
        List of PubMed IDs.
    """
    Entrez.email = email
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=max_results,
                            retmode='xml',
                            term=query)
    try:
        results = Entrez.read(handle)
    except Exception as e:
        return None
    return results['IdList']


def parse_quartile():
    """
    Get the quartile from the command line arguments.

    Returns:
    - email: str
    """

    quartile = parse_args().quartile

    return quartile


def parse_max_results():
    """
    Get the maximum number of results from the command line arguments.

    Returns:
    - max_results: int
    """

    max_results = parse_args().max_results

    return max_results


def reset_data():
    """
    Reset the data dictionary.

    Returns:
    - data: dict
    """
    data = {
        'Journal': [],
        'Year': [],
        'Month': [],
        'Type': [],
        'Pmid': [],
        'Doi': [],
        'Title': [],
        'Citations': [],
        'Abstract': [],
        'Disciplines': []
    }
    return data


def update_data(pubmed_id, data, metadata, disciplines):
    """
    Update the data dictionary with the metadata and disciplines.

    Parameters:
    - pubmed_id: str
    - data: dict
    - metadata: tuple
    - disciplines: str

    Returns:
    - data: dict
    """
    actual_journal, actual_year, month, title, publication_type, abstract, doi, num_citations = metadata
    data['Journal'].append(actual_journal)
    data['Year'].append(actual_year)
    data['Month'].append(month)
    data['Title'].append(title)
    data['Type'].append(publication_type)
    data['Pmid'].append(pubmed_id)
    data['Doi'].append(doi)
    data['Citations'].append(num_citations)
    data['Abstract'].append(abstract)
    data['Disciplines'].append(disciplines)

    return data


def save_data(data, output_file):
    """
    Save the data to the output file.

    Parameters:
    - data: dict
    - output_file: str
    """
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)


def check_alternate_journal_names(journal, journal_lut):
    """
    Check if the journal has an alternate name.

    Parameters:
    - journal: str
    - journal_lut: pandas.DataFrame

    Returns:
    - journal: str
    """
    if journal in journal_lut['Scimago'].values:
        journal = journal_lut.loc[journal_lut['Scimago'] == journal,
                                  'PubMed'].values[0]
    return journal
