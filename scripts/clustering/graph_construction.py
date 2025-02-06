import os
import igraph as ig
import psutil
from glob import glob
from src.utils.clustering import *
from src.utils.parsing import parse_directories

from src.utils.load_and_save import load_embedding_shards
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']

if __name__ == '__main__':
    configurations = load_configurations()['graph_construction']
    directories = parse_directories()

    shard_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['neuro'])
    graph_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['graphs'])

    files = glob(os.path.join(shard_directory, '*.h5'))
    graph_file = os.path.join(graph_directory, 'article_similarity.graphml')

    print('Loading embeddings...')
    embeddings, pmids = load_embedding_shards(files)

    num_points = embeddings.shape[0]
    num_neighbors = configurations['num_neighbors']
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Check if the selected k will fit into memory
    if not check_memory_constraints(num_points, num_neighbors,
                                    available_memory_gb):
        raise MemoryError(
            "Not enough memory for the selected k. Please reduce k or upgrade your hardware."
        )

    print('Constructing k-NN graph...')
    edges, weights = construct_knn_graph(embeddings, num_neighbors)

    print('Creating igraph Graph object...')
    G = ig.Graph(edges=edges, directed=False)
    G.vs['pmid'] = pmids
    G.es['weight'] = weights

    print('Saving graph...')
    os.makedirs(os.path.dirname(graph_file), exist_ok=True)
    G.write(graph_file, format='graphml')
