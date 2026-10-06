"""
Refetch CrossRef citation counts for all articles in the updated dataset (NeuroScape 2.0).

Counts are requested in batches of DOIs and appended to a JSON lines file together with the fetch
date, so the script can be interrupted and resumed. Citation rates are computed when the dataset is
assembled, relative to a fixed reference date.
"""

import os
import argparse
from tqdm import tqdm
from datetime import date

from src.utils.parsing import parse_directories
from src.utils.update import load_universe, fetch_crossref_works, append_jsonl, load_jsonl

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']
EMAIL = os.environ['EMAIL']

BATCH_SIZE = 100


def parse_args():
    """
    Parse the command line arguments.

    Returns:
    - args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Refetch CrossRef citation counts.')
    parser.add_argument('--output',
                        type=str,
                        default=f'citations_{date.today().isoformat()}.jsonl',
                        help='Output file name in the citations directory.')
    parser.add_argument('--limit',
                        type=int,
                        default=None,
                        help='Only process this many articles (dry run).')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    directories = parse_directories()

    citations_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['citations'])
    os.makedirs(citations_directory, exist_ok=True)
    output_file = os.path.join(citations_directory, args.output)

    articles = load_universe(BASEPATH, directories)
    if args.limit is not None:
        articles = articles.sample(args.limit, random_state=0)
    processed = {record['Pmid'] for record in load_jsonl(output_file)}
    articles = articles[~articles['Pmid'].isin(processed)]

    print(f'{len(processed)} articles already processed, '
          f'{len(articles)} to go.')

    for start in tqdm(range(0, len(articles), BATCH_SIZE)):
        batch = articles.iloc[start:start + BATCH_SIZE]
        dois = batch['Doi'].tolist()
        works = fetch_crossref_works(dois, 'DOI,is-referenced-by-count',
                                     EMAIL)
        fetched = date.today().isoformat()

        append_jsonl([{
            'Pmid': int(pubmed_id),
            'Doi': doi,
            'Citations': works[doi.lower()]['is-referenced-by-count']
            if doi.lower() in works else None,
            'Fetched': fetched
        } for pubmed_id, doi in zip(batch['Pmid'], dois)], output_file)
