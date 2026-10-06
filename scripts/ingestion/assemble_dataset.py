"""
Assemble the updated dataset (NeuroScape 2.0) from the v1 base, the previously processed delta and
the newly processed articles:

- article table with cluster assignments, the 'Added In' flag and (refreshed) citation rates
- domain embedding shards with updated in-links and out-links
- cluster table with updated sizes, citation rates and citation density statistics
- article citation graph and cluster citation density graph

Without --citations the assembly is provisional: v1 articles keep their v1 citation counts and
rates, other articles get none. With --citations, all counts come from the given refresh file and
ages and rates are computed relative to the reference date in config/ingestion/assembly.toml.
"""

import os
import json
import glob
import shutil
import argparse
import tomllib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.assembly import *
from src.utils.parsing import parse_directories
from src.utils.update import load_jsonl
from src.utils.load_and_save import load_articles_from_hdf5, save_articles_to_hdf5

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']

ARTICLE_COLUMNS = [
    'Pmid', 'Doi', 'Type', 'Title', 'Year', 'Month', 'Age', 'Citations',
    'Citation Rate', 'Cluster ID', 'Journal', 'Disciplines', 'Abstract',
    'Added In', 'Assignment Share', 'Assignment Margin', 'Citations Fetched'
]


def parse_args():
    """
    Parse the command line arguments.

    Returns:
    - args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description='Assemble the updated dataset.')
    parser.add_argument('--citations',
                        type=str,
                        default=None,
                        help='Citation refresh file in the citations directory.')
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='Output directory (default: the public data directory).')
    parser.add_argument('--link_pattern',
                        type=str,
                        default='link_candidates*.jsonl',
                        help='Link candidate files in the links directory.')

    return parser.parse_args()


def load_articles(directory):
    """
    Load all articles from a directory of HDF5 shards (in shard order).

    Parameters:
    - directory: str

    Returns:
    - articles: list of Article
    """

    articles = []
    for file_name in tqdm(sorted(glob.glob(os.path.join(directory, '*.h5')))):
        articles.extend(load_articles_from_hdf5(file_name, disable_tqdm=True))

    return articles


if __name__ == '__main__':
    args = parse_args()
    with open('config/ingestion/assembly.toml', 'rb') as f:
        configurations = tomllib.load(f)
    directories = parse_directories()

    base_directory = os.path.join(BASEPATH,
                                  directories['internal']['update']['base'])
    delta_directory = os.path.join(BASEPATH,
                                   directories['internal']['update']['delta'])
    new_hdf5_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['neuro'])
    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'],
        'Neuroscience')
    links_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['links'])
    citations_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['citations'])
    output_directory = args.output or os.path.join(BASEPATH, 'Public', 'Data')

    start_year = configurations['start_year']
    end_year = configurations['end_year']
    reference_date = configurations['reference_date']
    suffix = f'{start_year}-{end_year}'

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------
    print('Loading v1 articles...')
    base_articles = load_articles(
        os.path.join(base_directory, 'HDF5', 'DomainEmbeddings'))
    base_df = pd.read_csv(
        os.path.join(base_directory, 'CSV',
                     'neuroscience_articles_1999-2023.csv'))
    base_df['Added In'] = 'v1.0'
    base_df['Citations Fetched'] = 'v1.0'
    base_pmids = set(base_df['Pmid'])

    print('Loading added articles...')
    added_articles = load_articles(
        os.path.join(delta_directory, 'HDF5', 'Domain'))
    if os.path.isdir(new_hdf5_directory):
        added_articles += load_articles(new_hdf5_directory)
    added_articles = [
        article for article in added_articles
        if article.pmid not in base_pmids and article.year <= end_year
    ]
    added_articles = list({article.pmid: article
                           for article in added_articles}.values())
    added_pmids = [article.pmid for article in added_articles]

    metadata_files = [
        os.path.join(delta_directory, 'CSV', 'Neuroscience',
                     'articles_merged_cleaned_filtered.csv'),
        os.path.join(csv_directory, 'articles_merged_cleaned_filtered.csv')
    ]
    metadata_df = pd.concat([
        pd.read_csv(file_name) for file_name in metadata_files
        if os.path.exists(file_name)
    ])
    metadata_df = metadata_df.drop_duplicates(
        subset=['Pmid'], keep='last').set_index('Pmid')

    assignment_df = pd.read_csv(
        os.path.join(csv_directory, 'articles_assigned.csv')).set_index('Pmid')
    missing = set(added_pmids) - set(assignment_df.index)
    if missing:
        raise ValueError(f'{len(missing)} added articles have no cluster '
                         'assignment; run assign_to_clusters.py first.')

    added_df = metadata_df.loc[added_pmids, [
        'Doi', 'Type', 'Title', 'Year', 'Month', 'Journal', 'Disciplines',
        'Abstract'
    ]].reset_index()
    added_df['Year'] = added_df['Year'].astype(int)
    added_df = added_df.join(assignment_df, on='Pmid')
    added_df['Added In'] = np.where(added_df['Year'] >= 2024, 'v2.0',
                                    'v2.0-late')
    added_df['Citations'] = np.nan
    added_df['Citation Rate'] = np.nan
    added_df['Citations Fetched'] = None
    added_df['Age'] = compute_ages(added_df['Year'], added_df['Month'],
                                   reference_date)
    added_df = added_df.sort_values(['Year', 'Journal', 'Type'])

    article_df = pd.concat([base_df, added_df],
                           ignore_index=True)[ARTICLE_COLUMNS]
    articles = {
        article.pmid: article
        for article in base_articles + added_articles
    }
    print(f'Articles: {len(base_df)} (v1) + {len(added_df)} (added).')

    # ------------------------------------------------------------------
    # Citation counts and rates
    # ------------------------------------------------------------------
    if args.citations:
        citation_df = pd.DataFrame(
            load_jsonl(os.path.join(citations_directory, args.citations)))
        citation_df = citation_df.drop_duplicates(
            subset=['Pmid'], keep='last').set_index('Pmid')
        missing = set(article_df['Pmid']) - set(citation_df.index)
        if missing:
            raise ValueError(f'{len(missing)} articles have no refreshed '
                             'citation count; complete the refresh first.')
        article_df['Citations'] = citation_df.loc[article_df['Pmid'],
                                                  'Citations'].values
        article_df['Citations Fetched'] = citation_df.loc[article_df['Pmid'],
                                                          'Fetched'].values
        article_df['Age'] = compute_ages(article_df['Year'],
                                         article_df['Month'], reference_date)
        article_df['Citation Rate'] = article_df['Citations'] / article_df[
            'Age']
    else:
        print('No citation refresh given: provisional assembly.')

    # ------------------------------------------------------------------
    # Links
    # ------------------------------------------------------------------
    print('Resolving links...')
    in_links = {
        article.pmid: article.in_links
        for article in base_articles
    }
    out_links = {
        article.pmid: article.out_links
        for article in base_articles
    }
    v1_link_count = sum(len(links) for links in out_links.values())

    # Later candidate files (e.g. the January refetch) override earlier ones
    candidate_records = {}
    for file_name in sorted(
            glob.glob(os.path.join(links_directory, args.link_pattern))):
        candidate_records.update(
            (record['Pmid'], record) for record in load_jsonl(file_name))
    print(f'Link candidates for {len(candidate_records)} articles.')

    new_in_links, new_out_links = resolve_link_candidates(
        candidate_records.values(), article_df['Pmid'].values,
        article_df['Doi'].values)
    for pmid, links in new_in_links.items():
        in_links[pmid] = list(set(in_links.get(pmid, [])) | set(links))
    for pmid, links in new_out_links.items():
        out_links[pmid] = list(set(out_links.get(pmid, [])) | set(links))
    in_links, out_links = symmetrize_links(in_links, out_links)
    link_count = sum(len(links) for links in out_links.values())
    print(f'Citation links: {v1_link_count} (v1) -> {link_count}.')

    # ------------------------------------------------------------------
    # HDF5 shards
    # ------------------------------------------------------------------
    print('Saving articles...')
    hdf5_directory = os.path.join(output_directory, 'HDF5',
                                  'DomainEmbeddings')
    os.makedirs(hdf5_directory, exist_ok=True)
    for file_name in glob.glob(os.path.join(hdf5_directory, '*.h5')):
        os.remove(file_name)

    shard = []
    shard_id = 0
    items_per_shard = configurations['items_per_shard']
    for pmid, age, citations, citation_rate in tqdm(
            zip(article_df['Pmid'], article_df['Age'],
                article_df['Citations'], article_df['Citation Rate']),
            total=len(article_df)):
        article = articles[pmid]
        article.in_links = in_links.get(pmid, [])
        article.out_links = out_links.get(pmid, [])
        article.age = float(age)
        article.citation_count = -1 if pd.isna(citations) else int(citations)
        article.citation_rate = float(citation_rate)
        shard.append(article)
        if len(shard) == items_per_shard:
            save_articles_to_hdf5(shard,
                                  os.path.join(hdf5_directory,
                                               f'shard_{shard_id:04d}.h5'),
                                  disable_tqdm=True)
            shard = []
            shard_id += 1
    if shard:
        save_articles_to_hdf5(shard,
                              os.path.join(hdf5_directory,
                                           f'shard_{shard_id:04d}.h5'),
                              disable_tqdm=True)

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------
    print('Updating cluster statistics...')
    cluster_df = pd.read_csv(
        os.path.join(base_directory, 'CSV',
                     'neuroscience_clusters_1999-2023.csv'))
    cluster_ids = article_df['Cluster ID'].values.astype(int)
    num_clusters = cluster_ids.max() + 1

    grouped = article_df.groupby('Cluster ID')
    cluster_df['Size'] = cluster_df['Cluster ID'].map(grouped.size())
    cluster_df['Year First Article'] = cluster_df['Cluster ID'].map(
        grouped['Year'].min())
    for article_type in ['Research', 'Review']:
        medians = article_df[article_df['Type'] == article_type].groupby(
            'Cluster ID')['Citation Rate'].median()
        cluster_df[f'MCR {article_type}'] = cluster_df['Cluster ID'].map(
            medians)

    pmids = article_df['Pmid'].values
    node_df, weights = citation_density(
        pmids, cluster_ids, article_df['Age'].values,
        [out_links.get(pmid, []) for pmid in pmids],
        [in_links.get(pmid, []) for pmid in pmids], num_clusters)
    node_df = node_df.set_index('Cluster ID')
    for column in node_df.columns:
        cluster_df[column] = cluster_df['Cluster ID'].map(node_df[column])

    # ------------------------------------------------------------------
    # Save tables and graphs
    # ------------------------------------------------------------------
    print('Saving tables...')
    output_csv_directory = os.path.join(output_directory, 'CSV')
    os.makedirs(output_csv_directory, exist_ok=True)
    article_df.to_csv(os.path.join(output_csv_directory,
                                   f'neuroscience_articles_{suffix}.csv'),
                      index=False)
    cluster_df.to_csv(os.path.join(output_csv_directory,
                                   f'neuroscience_clusters_{suffix}.csv'),
                      index=False)
    # LLM-derived analyses are carried over from v1 unchanged
    for file_name in [
            'neuroscience_dimensions_1999-2023.csv', 'global_trends.csv',
            'just_trends.csv'
    ]:
        shutil.copyfile(os.path.join(base_directory, 'CSV', file_name),
                        os.path.join(output_csv_directory, file_name))

    print('Saving graphs...')
    graph_directory = os.path.join(output_directory, 'Graphs')
    os.makedirs(graph_directory, exist_ok=True)
    density_graph(weights, cluster_df['Cluster ID'].tolist()).save(
        os.path.join(graph_directory, 'cluster_citation_density.graphml'),
        format='graphml')
    article_citation_graph(pmids, cluster_ids, [
        out_links.get(pmid, []) for pmid in pmids
    ]).save(os.path.join(graph_directory, 'article_citation.graphml'),
            format='graphml')
    similarity_graph = os.path.join(graph_directory,
                                    'article_similarity.graphml')
    if not os.path.exists(similarity_graph):
        shutil.copyfile(
            os.path.join(base_directory, 'Graphs',
                         'article_similarity.graphml'), similarity_graph)

    models_directory = os.path.join(output_directory, 'Models')
    os.makedirs(models_directory, exist_ok=True)
    for file_name in os.listdir(os.path.join(base_directory, 'Models')):
        shutil.copyfile(os.path.join(base_directory, 'Models', file_name),
                        os.path.join(models_directory, file_name))

    report = {
        'version': configurations['version'],
        'provisional': args.citations is None,
        'reference_date': reference_date,
        'citations_file': args.citations,
        'articles': article_df['Added In'].value_counts().to_dict(),
        'articles_per_year': {
            int(year): int(count)
            for year, count in article_df['Year'].value_counts().sort_index().
            items()
        },
        'citation_links': {
            'v1': v1_link_count,
            'total': link_count
        },
        'link_candidate_articles': len(candidate_records),
    }
    with open(os.path.join(output_directory, 'assembly_report.json'),
              'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
