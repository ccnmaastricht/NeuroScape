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
    parser.add_argument('--start_year',
                        type=int,
                        default=1999,
                        help='Starting year for scraping articles.')
    parser.add_argument('--end_year',
                        type=int,
                        default=2023,
                        help='Ending year for scraping articles.')

    return parser.parse_args()


def parse_discipline():
    """
    Get the discipline from the command line arguments.    

    Returns:
    - discipline: str
    """

    discipline = parse_args().discipline

    return discipline


def parse_quartile():
    """
    Get the quartile from the command line arguments.    

    Returns:
    - quartile: str
    """

    quartile = parse_args().quartile

    return quartile


def parse_end_year():
    """
    Get the end year from the command line arguments.  

    Returns:
    - year: int
    """

    year = parse_args().end_year

    return year


def parse_start_year():
    """
    Get the start year from the command line arguments.    

    Returns:
    - start_year: int
    """

    start_year = parse_args().start_year

    return start_year


def parse_max_results():
    """
    Get the maximum number of results from the command line arguments.    

    Returns:
    - max_results: int
    """

    max_results = parse_args().max_results

    return max_results
