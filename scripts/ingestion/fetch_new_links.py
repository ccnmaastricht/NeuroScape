"""
Fetch citation link candidates for articles added in the dataset update (NeuroScape 2.0).

For every new article, the PubMed IDs of citing articles (PubMed "cited in") and the DOIs of its
references (CrossRef) are stored unfiltered in a JSON lines file. Intersecting them with the
articles in the dataset happens when the dataset is assembled, so the candidates stay valid if the
set of articles changes. The script is resumable and requests are batched.
"""

import os
import socket
import argparse
from time import sleep
from tqdm import tqdm
from datetime import date
from Bio import Entrez

from src.utils.adjacency import load_configurations
from src.utils.parsing import parse_directories
from src.utils.update import load_universe, fetch_crossref_works, append_jsonl, load_jsonl

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']
EMAIL = os.environ['EMAIL']

# Fail (and retry) instead of hanging on stalled network requests
socket.setdefaulttimeout(120)

PUBMED_BATCH_SIZE = 200
CROSSREF_BATCH_SIZE = 100


def parse_args():
    """
    Parse the command line arguments.

    Returns:
    - args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Fetch citation link candidates for new articles.')
    parser.add_argument('--min_year',
                        type=int,
                        default=None,
                        help='Only articles from this year on (e.g. for the January refetch).')
    parser.add_argument('--output',
                        type=str,
                        default='link_candidates.jsonl',
                        help='Output file name in the links directory.')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Only process this many articles (dry run).')

    return parser.parse_args()


def fetch_cited_in(pubmed_ids):
    """
    Fetch the PubMed IDs of articles citing each of the given articles (one linkset per article).

    Parameters:
    - pubmed_ids: list of int

    Returns:
    - cited_in: dict mapping PubMed ID to list of citing PubMed IDs
    """

    handle = Entrez.elink(dbfrom='pubmed',
                          id=[str(pubmed_id) for pubmed_id in pubmed_ids],
                          linkname='pubmed_pubmed_citedin')
    records = Entrez.read(handle)
    handle.close()

    cited_in = {}
    for record in records:
        pubmed_id = int(record['IdList'][0])
        links = record['LinkSetDb'][0]['Link'] if record['LinkSetDb'] else []
        cited_in[pubmed_id] = [int(link['Id']) for link in links]

    return cited_in


def fetch_references(dois):
    """
    Fetch the DOIs of the references of each of the given articles.

    Parameters:
    - dois: list of str

    Returns:
    - references: dict mapping lower-case DOI to list of reference DOIs (None if not found)
    """

    works = fetch_crossref_works(dois, 'DOI,reference', EMAIL)

    references = {}
    for doi in dois:
        work = works.get(doi.lower())
        references[doi.lower()] = None if work is None else [
            reference['DOI'] for reference in work.get('reference', [])
            if 'DOI' in reference
        ]

    return references


def with_retries(function, arguments, num_attempts, sleep_time):
    """
    Call a function, retrying on failure.
    """

    for attempt in range(num_attempts):
        try:
            return function(arguments)
        except Exception as error:
            if attempt == num_attempts - 1:
                raise
            print(f'Retrying after error: {error}')
            sleep(sleep_time)


if __name__ == '__main__':
    args = parse_args()
    configurations = load_configurations()['pubmed_requests']
    num_attempts = configurations['num_attempts']
    sleep_time = configurations['sleep_time']
    Entrez.email = EMAIL
    if 'NCBI_API_KEY' in os.environ:
        Entrez.api_key = os.environ['NCBI_API_KEY']

    directories = parse_directories()
    links_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['links'])
    os.makedirs(links_directory, exist_ok=True)
    output_file = os.path.join(links_directory, args.output)

    universe = load_universe(BASEPATH, directories)
    articles = universe[universe['Source'] != 'base']
    if args.min_year is not None:
        articles = articles[articles['Year'] >= args.min_year]

    processed = {record['Pmid'] for record in load_jsonl(output_file)}
    articles = articles[~articles['Pmid'].isin(processed)]
    if args.limit is not None:
        articles = articles.head(args.limit)

    print(f'{len(processed)} articles already processed, '
          f'{len(articles)} to go.')

    fetched = date.today().isoformat()
    for start in tqdm(range(0, len(articles), CROSSREF_BATCH_SIZE)):
        batch = articles.iloc[start:start + CROSSREF_BATCH_SIZE]
        pubmed_ids = batch['Pmid'].astype(int).tolist()
        dois = batch['Doi'].tolist()

        cited_in = {}
        for i in range(0, len(pubmed_ids), PUBMED_BATCH_SIZE):
            cited_in.update(
                with_retries(fetch_cited_in,
                             pubmed_ids[i:i + PUBMED_BATCH_SIZE],
                             num_attempts, sleep_time))
        references = with_retries(fetch_references, dois, num_attempts,
                                  sleep_time)

        append_jsonl([{
            'Pmid': pubmed_id,
            'Doi': doi,
            'Cited In': cited_in.get(pubmed_id, []),
            'References': references[doi.lower()],
            'Fetched': fetched
        } for pubmed_id, doi in zip(pubmed_ids, dois)], output_file)
