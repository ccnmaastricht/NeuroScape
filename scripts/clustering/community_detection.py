import os
import leidenalg
import numpy as np
import pandas as pd
import igraph as ig
from collections import deque

from src.utils.clustering import load_configurations
from src.utils.parsing import parse_directories

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']

if __name__ == '__main__':

    configurations = load_configurations()
    directories = parse_directories()

    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'],
        'Neuroscience')
    graph_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['graphs'])

    csv_file = os.path.join(csv_directory, 'merged_filtered.csv')

    graph_file = os.path.join(graph_directory, 'article_similarity.graphml')

    print('loading graph')
    G = ig.Graph.Read(graph_file, format='graphml')
    pmids = G.vs['pmid']

    print(
        'running Leiden community detection for different resolution parameters'
    )
    num_resolution_parameter = configurations['num_resolution_parameter']
    max_resolution_parameter = configurations['max_resolution_parameter']
    min_resolution_parameter = max_resolution_parameter / num_resolution_parameter

    resolution_parameters = np.linspace(min_resolution_parameter,
                                        max_resolution_parameter,
                                        num_resolution_parameter)

    modularity_values = np.zeros(num_resolution_parameter)
    num_unique_clusters = np.zeros(num_resolution_parameter)

    decreasing = deque(np.zeros(5, dtype=bool))

    for i, resolution_parameter in enumerate(resolution_parameters):
        partition = leidenalg.find_partition(
            G,
            leidenalg.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution_parameter)

        modularity_values[i] = G.modularity(partition.membership,
                                            weights='weight')
        num_unique_clusters[i] = len(np.unique(partition.membership))
        print(f'number of unique clusters: {num_unique_clusters[i]}')
        print(f'modularity values: {modularity_values[i]}')

        if i > 0:
            decreasing.popleft()
            decreasing.append(modularity_values[i] < modularity_values[i - 1])

        if all(decreasing):
            print('modularity is decreasing, breaking')
            break

    best_resolution_parameter = resolution_parameters[np.argmax(
        modularity_values)]
    partition = leidenalg.find_partition(
        G,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=best_resolution_parameter)

    pmid_cluster = dict(zip(pmids, partition.membership))

    print('saving cluster')
    df = pd.read_csv(csv_file)
    df['cluster'] = df['Pmid'].map(pmid_cluster)

    new_csf_file_name = configurations['csv_file_name'].replace(
        '.csv', '_clustered.csv')
    new_csv_file = os.path.join(csv_directory, new_csf_file_name)

    df.to_csv(new_csv_file, index=False)
