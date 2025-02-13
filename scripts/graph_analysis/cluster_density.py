import os
import pandas as pd
import igraph as ig
from glob import glob
from tqdm import tqdm
from src.utils.cluster_graph import *
from src.utils.parsing import parse_directories
from src.utils.load_and_save import load_articles_from_hdf5

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']

if __name__ == '__main__':
    configurations = load_configurations()
    directories = parse_directories()

    print('Loading csv files...')

    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'],
        'Neuroscience')

    article_csv_file = os.path.join(
        csv_directory, 'articles_merged_cleaned_filtered_clustered.csv')

    cluster_csv_file = os.path.join(
        csv_directory,
        'clusters_defined_distinguished_questions_trends_assessed.csv')

    graph_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['graphs'])
    graph_file = os.path.join(graph_directory, 'citation_density.graphml')
    article_df = pd.read_csv(article_csv_file)
    cluster_df = pd.read_csv(cluster_csv_file)

    print('Loading articles...')
    shard_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['neuro'])
    article_files = glob(os.path.join(shard_directory, '*.h5'))

    article_graph = {
        'Pmid': [],
        'Cluster ID': [],
        'Year': [],
        'Age': [],
        'In_links': [],
        'Out_links': []
    }

    for file in tqdm(article_files):
        articles = load_articles_from_hdf5(file, disable_tqdm=True)
        for article in articles:
            article_graph['Pmid'].append(article.pmid)
            article_graph['Cluster ID'].append(
                article_df.loc[article_df['Pmid'] == article.pmid,
                               'Cluster ID'].values[0])
            article_graph['Year'].append(article.year)
            article_graph['Age'].append(article.age)
            article_graph['In_links'].append(article.in_links)
            article_graph['Out_links'].append(article.out_links)

    article_graph_df = pd.DataFrame(article_graph)

    cluster_df['Reference Krackhardt'] = 0.0
    cluster_df['Citation Krackhardt'] = 0.0
    cluster_df['Most Cited Cluster'] = ''
    cluster_df['Most Citing Cluster'] = ''

    edge_list = []
    weights_list = []

    print('Performing density analyses...')
    for source_cluster in tqdm(cluster_df['Cluster ID']):
        krackhardt, frequent_clusters = node_analysis(article_graph_df,
                                                      source_cluster)
        cluster_df.loc[cluster_df['Cluster ID'] == source_cluster,
                       'Reference Krackhardt'] = krackhardt['References']
        cluster_df.loc[cluster_df['Cluster ID'] == source_cluster,
                       'Citation Krackhardt'] = krackhardt['Citations']
        cluster_df.loc[cluster_df['Cluster ID'] == source_cluster,
                       'Most Cited Cluster'] = frequent_clusters['References']
        cluster_df.loc[cluster_df['Cluster ID'] == source_cluster,
                       'Most Citing Cluster'] = frequent_clusters['Citations']

        for destination_cluster in cluster_df['Cluster ID']:
            edge = tuple([source_cluster, destination_cluster])
            edge_list.append(edge)
            weight = edge_analysis(article_graph_df, edge)
            weights_list.append(weight)

    # Most Cited Cluster and Most Citing Cluster before Most Similar Cluster
    columns = cluster_df.columns.tolist()
    columns = columns[:6] + columns[-4:] + columns[6:-4]
    cluster_df = cluster_df[columns]

    print('Saving csv file...')
    cluster_csv_file = cluster_csv_file.replace('.csv', '_density.csv')
    cluster_df.to_csv(cluster_csv_file, index=False)

    print('Creating igraph Graph object')
    reference_density_graph = ig.Graph(edges=edge_list, directed=True)
    reference_density_graph.es['weight'] = weights_list
    reference_density_graph.vs['label'] = cluster_df['Cluster ID']

    print('Saving graphs...')
    reference_density_graph.save(graph_file, format='graphml')
