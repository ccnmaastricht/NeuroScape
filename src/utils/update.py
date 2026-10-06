import os
import json
import glob
import time
import h5py
import urllib.parse
import urllib.request
import pandas as pd

CROSSREF_URL = 'https://api.crossref.org/works'


def normalize_doi(dois):
    """
    Normalize DOIs for matching (DOIs are case-insensitive).

    Parameters:
    - dois: pd.Series

    Returns:
    - dois: pd.Series
    """

    return dois.astype(str).str.strip().str.lower()


def load_known_from_hdf5(directory):
    """
    Load the PubMed IDs, DOIs and years of all articles stored in a directory of HDF5 shards.

    Parameters:
    - directory: str

    Returns:
    - known: pd.DataFrame with columns 'Pmid', 'Doi' and 'Year'
    """

    pmids, dois, years = [], [], []
    for file_name in glob.glob(os.path.join(directory, '*.h5')):
        with h5py.File(file_name, 'r') as file:
            pmids.extend(file['pmid'][:].tolist())
            dois.extend(doi.decode() if isinstance(doi, bytes) else doi
                        for doi in file['doi'][:])
            years.extend(file['year'][:].tolist())

    return pd.DataFrame({'Pmid': pmids, 'Doi': dois, 'Year': years})


def load_universe(basepath, directories):
    """
    Load PubMed IDs, DOIs and years of all articles in the updated dataset: the v1 base,
    the previously processed delta and the newly processed articles.

    Parameters:
    - basepath: str
    - directories: dict

    Returns:
    - universe: pd.DataFrame with columns 'Pmid', 'Doi', 'Year' and 'Source'
    """

    base_directory = os.path.join(basepath,
                                  directories['internal']['update']['base'])
    delta_directory = os.path.join(basepath,
                                   directories['internal']['update']['delta'])
    new_directory = os.path.join(
        basepath, directories['internal']['intermediate']['hdf5']['neuro'])

    base_df = pd.read_csv(os.path.join(base_directory, 'CSV',
                                       'neuroscience_articles_1999-2023.csv'),
                          usecols=['Pmid', 'Doi', 'Year'])
    base_df['Source'] = 'base'
    delta_df = load_known_from_hdf5(
        os.path.join(delta_directory, 'HDF5', 'Domain'))
    delta_df['Source'] = 'delta'
    new_df = load_known_from_hdf5(new_directory)
    new_df['Source'] = 'new'

    universe = pd.concat([base_df, delta_df, new_df], ignore_index=True)
    universe = universe.drop_duplicates(subset=['Pmid'], keep='first')

    return universe


def crossref_request(params, email, num_attempts=5, sleep_time=10):
    """
    Query the CrossRef works endpoint (polite pool).

    Parameters:
    - params: dict
    - email: str
    - num_attempts: int
    - sleep_time: float

    Returns:
    - items: list of dict
    """

    url = f'{CROSSREF_URL}?{urllib.parse.urlencode(params)}'
    request = urllib.request.Request(
        url, headers={'User-Agent': f'NeuroScape (mailto:{email})'})

    for attempt in range(num_attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)['message']['items']
        except Exception:
            if attempt == num_attempts - 1:
                raise
            time.sleep(sleep_time)


def fetch_crossref_works(dois, select, email):
    """
    Fetch CrossRef metadata for a batch of DOIs (at most 100).

    Parameters:
    - dois: list of str
    - select: str, comma-separated CrossRef fields
    - email: str

    Returns:
    - works: dict mapping lower-case DOI to the CrossRef item
    """

    params = {
        'filter': ','.join(f'doi:{doi}' for doi in dois),
        'rows': len(dois),
        'select': select
    }
    items = crossref_request(params, email)

    return {item['DOI'].lower(): item for item in items}


def append_jsonl(records, file_name):
    """
    Append records to a JSON lines file.

    Parameters:
    - records: list of dict
    - file_name: str
    """

    with open(file_name, 'a') as file:
        for record in records:
            file.write(json.dumps(record) + '\n')


def load_jsonl(file_name):
    """
    Load a JSON lines file.

    Parameters:
    - file_name: str

    Returns:
    - records: list of dict
    """

    if not os.path.exists(file_name):
        return []

    with open(file_name) as file:
        return [json.loads(line) for line in file if line.strip()]
