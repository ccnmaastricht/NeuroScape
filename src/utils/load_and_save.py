import os
import h5py
import glob
import numpy as np

from tqdm import tqdm

from src.classes.data_types import Article


def determine_output_filename(output_folder, file_extension):
    """
    Determine the output filename.

    Parameters:
    - output_folder: str
    - file_extension: str

    Returns:
    - output_file: str
    - shard_id: int
    """
    existing_shards = glob.glob(
        os.path.join(output_folder, f'*.{file_extension}'))
    num_shards = len(existing_shards)
    shard_id = num_shards + 1
    return os.path.join(output_folder,
                        f'shard_{shard_id:04d}.{file_extension}'), shard_id


def save_articles_to_hdf5(articles, filename, disable_tqdm=False):
    """
    Save the articles to an HDF5 file.

    Parameters:
    - articles: list of Article
    - filename: str
    - disable_tqdm: bool
    """
    with h5py.File(filename, 'w') as file:
        # Create datasets for the attributes that are same across all entries
        pmids = file.create_dataset('pmid', (len(articles), ), dtype='i')
        dois = file.create_dataset('doi', (len(articles), ),
                                   dtype=h5py.string_dtype())
        titles = file.create_dataset('title', (len(articles), ),
                                     dtype=h5py.string_dtype())
        types = file.create_dataset('type', (len(articles), ),
                                    dtype=h5py.string_dtype())
        journals = file.create_dataset('journal', (len(articles), ),
                                       dtype=h5py.string_dtype())
        years = file.create_dataset('year', (len(articles), ), dtype='i')
        ages = file.create_dataset('age', (len(articles), ), dtype='f')
        citation_counts = file.create_dataset('citation_count',
                                              (len(articles), ),
                                              dtype='i')
        citation_rates = file.create_dataset('citation_rate',
                                             (len(articles), ),
                                             dtype='f')
        abstracts = file.create_dataset('abstract', (len(articles), ),
                                        dtype=h5py.string_dtype())

        # Create a group for embeddings and adjacencies to handle variability in lengths
        embedding_group = file.create_group('embeddings')
        in_group = file.create_group('in_links')
        out_group = file.create_group('out_links')

        # Populate the datasets
        for i, article in tqdm(enumerate(articles),
                               total=len(articles),
                               disable=disable_tqdm):

            pmids[i] = article.pmid
            dois[i] = article.doi
            titles[i] = article.title
            types[i] = article.type
            journals[i] = article.journal
            years[i] = article.year
            ages[i] = article.age
            citation_counts[i] = article.citation_count
            citation_rates[i] = article.citation_rate
            abstracts[i] = article.abstract

            # Handle possibly empty embeddings and adjacencies
            if np.array(article.embedding).size > 0:
                embedding_group.create_dataset(str(i),
                                               data=np.array(
                                                   article.embedding),
                                               dtype='f')
            else:
                embedding_group.create_dataset(str(i), shape=(0, ), dtype='f')

            if np.array(article.in_links).size > 0:
                in_group.create_dataset(str(i),
                                        data=np.array(article.in_links),
                                        dtype='i')
            else:
                in_group.create_dataset(str(i), shape=(0, ), dtype='i')

            if np.array(article.out_links).size > 0:
                out_group.create_dataset(str(i),
                                         data=np.array(article.out_links),
                                         dtype='i')
            else:
                out_group.create_dataset(str(i), shape=(0, ), dtype='i')


def load_articles_from_hdf5(filename, disable_tqdm=False):
    """
    Load articles from an HDF5 file and return a list of Article objects.

    Parameters:
    - filename: str
    - disable_tqdm: bool

    Returns:
    - List[Article]: List of articles loaded from the file.
    """
    articles = []
    with h5py.File(filename, 'r') as file:
        # Retrieve datasets
        pmids = file['pmid'][:]
        dois = file['doi'][:]
        titles = file['title'][:]
        types = file['type'][:]
        journals = file['journal'][:]
        years = file['year'][:]
        ages = file['age'][:]
        citation_counts = file['citation_count'][:]
        citation_rates = file['citation_rate'][:]
        abstracts = file['abstract'][:]

        # Retrieve embeddings, in_links, and out_links from their groups
        embedding_group = file['embeddings']
        in_group = file['in_links']
        out_group = file['out_links']

        total_articles = len(pmids)

        for i in tqdm(range(total_articles),
                      total=total_articles,
                      disable=disable_tqdm):
            embedding = np.array(embedding_group[str(i)])
            in_links = np.array(in_group[str(i)])
            out_links = np.array(out_group[str(i)])

            # Instantiate Article with data retrieved
            article = Article(pmid=int(pmids[i]),
                              doi=dois[i].decode(),
                              title=titles[i].decode(),
                              type=types[i].decode(),
                              journal=journals[i].decode(),
                              year=int(years[i]),
                              age=float(ages[i]),
                              citation_count=int(citation_counts[i]),
                              citation_rate=float(citation_rates[i]),
                              abstract=abstracts[i].decode(),
                              embedding=embedding.tolist(),
                              in_links=in_links.tolist(),
                              out_links=out_links.tolist())

            articles.append(article)

    return articles


def load_embeddings_from_hdf5(filename):
    """
    Load embeddings from an HDF5 file (single shard) and return a numpy array of embeddings. Also return the associated PMIDs.

    Parameters:
    - filename: str

    Returns:
    - embeddings: np.array
    - pmids: List of integers
    """

    with h5py.File(filename, 'r') as file:

        # Retrieve pmids to compute total number of articles
        pmids = file['pmid'][:]
        total_articles = len(pmids)

        # Retrieve embeddings
        embedding_group = file['embeddings']

        # Initialize embeddings array with zeros
        first_embedding = np.array(embedding_group[str(0)])
        embeddings = np.zeros((total_articles, first_embedding.shape[0]))

        # Assign embeddings
        for i in range(total_articles):
            embeddings[i] = np.array(embedding_group[str(i)])

    return embeddings, pmids


def load_embedding_shards(embeddings_files, disable_tqdm=False):
    """
    Load embeddings from multiple shards and concatenate them into a single array. Also return the associated PMIDs.

    Parameters:
    - embeddings_files: List of strings, the paths to the embeddings files.
    - disable_tqdm: bool

    Returns:
    - embeddings: np.array, the concatenated embeddings.
    - pmids: List of integers, the PMIDs of the articles.
    """

    embeddings = []
    pmids = []

    for file in tqdm(embeddings_files, disable=disable_tqdm):
        embeddings_shard, pmids_shard = load_embeddings_from_hdf5(file)
        embeddings.append(embeddings_shard)
        pmids.extend(pmids_shard)

    return np.concatenate(embeddings, axis=0), pmids


def align_to_df(embeddings, pmids, df):
    """
    Aligns the order of embeddings and PMIDs to the PMID order in the DataFrame.

    Parameters:
    embeddings (np.ndarray): The embeddings to align.
    pmids (np.ndarray): The PMIDs to align.
    df (pd.DataFrame): The DataFrame to align to.

    Returns:
    aligned_embeddings (np.ndarray): The aligned embeddings.
    aligned_pmids (np.ndarray): The aligned PMIDs.
    """
    pmids = np.array(pmids)
    pmid_to_index = {pmid: idx for idx, pmid in enumerate(pmids)}
    ordered_indices = [pmid_to_index[pmid] for pmid in df['Pmid'].values]
    return embeddings[ordered_indices], pmids[ordered_indices]
