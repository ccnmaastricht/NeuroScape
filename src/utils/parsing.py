import tomllib
import argparse


def parse_directories():
    """
    Parse the directories from the configuration file.
    Returns:
    - directories: dict
    """
    with open('config/directories.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def parse_args():
    """
    Parse the command line arguments.
    
    Returns:
    - args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Scrape articles from PubMed.')
    parser.add_argument('--discipline',
                        type=str,
                        default='Neuroscience',
                        help='Discipline to scrape articles for.')
    parser.add_argument('--quartile',
                        type=str,
                        default='Q1',
                        help='Quartile to scrape articles for.')
    parser.add_argument('--max_results',
                        type=int,
                        default=5000,
                        help='Maximum number of results to return.')

    return parser.parse_args()


def parse_discipline():
    """
    Get the discipline from the command line arguments.    

    Returns:
    - discipline: str
    """

    discpline = parse_args().discipline

    return discpline
