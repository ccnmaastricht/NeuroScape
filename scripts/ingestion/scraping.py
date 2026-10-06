"""
This script is designed to automate the process of scraping journal article metadata and abstracts from PubMed.
"""

import os
import glob
import pandas as pd
from time import sleep

from src.utils.parsing import *
from src.utils.checkpoints import *
from src.utils.scraping import *
from src.utils.load_and_save import determine_output_filename
from src.classes.article_metadata import ArticleMetadata

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']
EMAIL = os.environ['EMAIL']


def build_areas_lookup(input_files):
    """
    Map journal titles to Scimago subject areas, using the most recent file that has an 'Areas' column.
    Recent Scimago exports (2025 onwards) no longer contain this column.

    Parameters:
    - input_files: list of str

    Returns:
    - areas_lookup: dict
    """
    areas_lookup = {}
    for file in sorted(input_files):
        scimago_df = pd.read_csv(file, sep=';')
        if 'Areas' in scimago_df.columns:
            areas_lookup.update(zip(scimago_df['Title'], scimago_df['Areas']))

    return areas_lookup


if __name__ == '__main__':

    # Initialize the ArticleMetadata
    articlemetadata = ArticleMetadata()

    # Load the configurations and unpack the scraping parameters
    directories = parse_directories()
    scraping_parameters = load_configurations()
    prefix, suffix, sleep_time, num_attempts, items_per_shard = unpack_scraping_parameters(
        scraping_parameters)
    discipline = parse_discipline()
    quartile = parse_quartile()
    max_results = parse_max_results()
    start_year = parse_start_year()
    end_year = parse_end_year()

    print(f'Scraping data for {discipline}...')

    # Define the base directory and the input, processed, and output folders
    input_folder = os.path.join(
        BASEPATH, directories['internal']['reference']['scimago'], discipline)
    input_files = glob.glob(os.path.join(input_folder, "*.csv"))
    areas_lookup = build_areas_lookup(input_files)
    input_files = [
        file for file in input_files if start_year <= int(
            os.path.basename(file).split(prefix)[1].split(suffix)[0]) <= end_year
    ]

    checkpoints_folder = os.path.join(BASEPATH,
                                      directories['internal']['checkpoints'])
    lut_file = os.path.join(BASEPATH,
                            directories['internal']['reference']['reference'],
                            'journal_lut.csv')

    output_folder = os.path.join(BASEPATH, directories['internal']['raw'],
                                 discipline)
    os.makedirs(output_folder, exist_ok=True)

    # Determine the output file
    output_file, shard_id = determine_output_filename(output_folder, 'csv')

    # Initialize the data dictionary
    data = reset_data()

    # Load the lookup table for relating Scimago and PubMed journal names
    lut = pd.read_csv(lut_file)

    # Load the processed articles
    processed_file = os.path.join(checkpoints_folder, 'scraped_articles.json')
    processed_articles = load_processed_articles(processed_file)

    # Initialize the number of items in the current shard
    num_items = 0
    print('Searching for articles...')

    # Loop through each input (scimago) file
    for file in sorted(input_files):
        year = file.split(prefix)[1].split(suffix)[0]
        print(f' Year: {year}')

        # Load the dataframe from the input file
        scimago_df = pd.read_csv(file, sep=';')
        scimago_df = scimago_df[scimago_df['SJR Best Quartile'] == quartile]

        # Loop through each journal falling within the specified quartile
        for journal in scimago_df['Title']:
            print(f' Journal: {journal}')

            # Get the disciplines this journal falls under
            if 'Areas' in scimago_df.columns:
                areas = scimago_df[scimago_df['Title'] ==
                                   journal]['Areas'].values[0]
            else:
                areas = areas_lookup.get(journal, discipline)
            disciplines = areas.replace(';', ' /')

            # Check if the journal has an alternate name
            query_journal = check_alternate_journal_names(journal, lut)

            # Define the PubMed query
            query = f"""
            ("{query_journal}"[Journal]) AND  (("{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication]))
            """

            # Try to get the PubMed IDs for the query
            pubmed_ids = None
            for _ in range(num_attempts):
                try:
                    pubmed_ids = get_id_list(query,
                                             EMAIL,
                                             max_results=max_results)
                    break
                except:
                    sleep(sleep_time)

            # If no PubMed IDs were found, continue to the next journal
            if pubmed_ids is None:
                continue

            # Initialize the number of obtained articles
            total_articles = len(pubmed_ids)
            obtained_articles = 0
            print(f' Total articles found: {total_articles}')

            # Loop through each PubMed ID
            for article_id in pubmed_ids:

                # If the article has already been processed, continue to the next article
                if article_id in processed_articles:
                    obtained_articles += 1
                    continue

                # Add the article to the set of processed articles
                processed_articles.add(article_id)

                # Try to fetch the metadata for the article
                metadata = None
                for _ in range(num_attempts):
                    try:
                        metadata = articlemetadata.fetch(article_id)
                        break
                    except:
                        sleep(sleep_time)

                # If metadata was found, add it to the data dictionary
                if metadata is not None:
                    data = update_data(article_id, data, metadata, disciplines)
                    obtained_articles += 1
                    num_items += 1

                # If the number of items in the current shard is equal to the items per shard,
                # save the data and update the output file (new shard)
                if (num_items == items_per_shard):
                    save_data(data, output_file)
                    save_processed_articles(processed_file, processed_articles)

                    data = reset_data()
                    shard_id = shard_id + 1
                    output_file = os.path.join(output_folder,
                                               f'shard_{shard_id:04d}.csv')
                    num_items = 0
            print(f' Articles obtained: {obtained_articles}')

    # Save the last (partial) shard
    if num_items > 0:
        save_data(data, output_file)
    save_processed_articles(processed_file, processed_articles)
