import os

import glob
import pickle
import tomllib
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm

from src.utils.filtering import *
from src.utils.parsing import parse_directories
from src.utils.classifier import get_disciplines
from src.utils.load_and_save import save_articles_to_hdf5

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']


def filter(embedding_files, dataframe, device, model, class_index, cutoff):
    """
    Filter the data based on the filter network.

    Parameters:
    - embedding_files: list
    - dataframe: pd.DataFrame
    - device: str
    - model: DisciplineClassifier
    - class_index: int
    - cutoff: float

    Returns:
    - filtered_data: list
    - dataframe: pd.DataFrame
    """

    global_keep_index = []
    filtered_data = []
    for file in tqdm(embedding_files, total=len(embedding_files)):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        pubmed_ids = data.pmids
        embeddings = data.embeddings

        embeddings_on_device = torch.tensor(embeddings).to(device)
        probabilities = model(embeddings_on_device).cpu().detach().numpy()

        class_probabilities = probabilities[:, class_index]
        max_probabilities = np.max(probabilities, axis=1)
        confidence = class_probabilities / max_probabilities

        disciplines = get_disciplines(dataframe, pubmed_ids)
        remove_indices = remove(disciplines, confidence, cutoff)

        nan_indices = np.isnan(embeddings).any(axis=1)
        inf_indices = np.isinf(embeddings).any(axis=1)

        keep_index = [
            pubmed_ids[i] for i in range(len(pubmed_ids)) if
            not remove_indices[i] and not nan_indices[i] and not inf_indices[i]
        ]

        global_keep_index = update_keep_index(global_keep_index, keep_index,
                                              dataframe)

        for pmid in keep_index:
            index = pubmed_ids.index(pmid)
            article = fill_article(pmid, dataframe, embeddings[index])
            filtered_data.append(article)

    drop_index = list(set(dataframe.index) - set(global_keep_index))
    dataframe = dataframe.drop(drop_index)

    return filtered_data, dataframe


if __name__ == '__main__':
    configurations = load_configurations()
    items_per_shard = configurations['filtering']['shard_size']
    neuro_class_index = configurations['filtering']['class_index']
    confidence_cutoff = configurations['filtering']['confidence_cutoff']

    data_directories = parse_directories()

    model_file = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['models'],
        'discipline_classification_model_finetuned.pth')

    model = load_model(configurations['model'], model_file)
    device = configurations['model']['device']

    multidisciplinary_dataframe_dir = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['csv'],
        'Multidisciplinary')
    neuroscience_dataframe_dir = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['csv'],
        'Neuroscience')

    multidisciplinary_embedding_dir = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['embeddings'],
        'Multidisciplinary')

    neuroscience_embedding_dir = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['embeddings'],
        'Neuroscience')

    filtered_data = []

    multi_dataframe = pd.read_csv(
        os.path.join(multidisciplinary_dataframe_dir,
                     'articles_merged_cleaned.csv'))
    neuro_dataframe = pd.read_csv(
        os.path.join(neuroscience_dataframe_dir,
                     'articles_merged_cleaned.csv'))

    multi_embedding_files = glob.glob(
        os.path.join(multidisciplinary_embedding_dir, '*.pkl'))
    neuro_embedding_files = glob.glob(
        os.path.join(neuroscience_embedding_dir, '*.pkl'))

    print('Filtering multidisciplinary data...')

    multi_filtered_data, multi_dataframe = filter(multi_embedding_files,
                                                  multi_dataframe, device,
                                                  model, neuro_class_index,
                                                  confidence_cutoff)

    print('Filtering neuroscience data...')
    neuro_filtered_data, neuro_dataframe = filter(neuro_embedding_files,
                                                  neuro_dataframe, device,
                                                  model, neuro_class_index,
                                                  confidence_cutoff)

    print('Merging data...')
    dataframe = pd.concat([multi_dataframe, neuro_dataframe],
                          ignore_index=True)

    filtered_data.extend(multi_filtered_data)
    filtered_data.extend(neuro_filtered_data)

    output_directory = os.path.join(
        BASEPATH,
        data_directories['internal']['intermediate']['hdf5']['voyage'])

    os.makedirs(output_directory, exist_ok=True)

    df_output_file = os.path.join(neuroscience_dataframe_dir,
                                  'articles_merged_cleaned_filtered.csv')
    emb_output_file = os.path.join(output_directory,
                                   'articles_merged_cleaned_filtered.h5')

    print('Saving data...')
    dataframe.to_csv(df_output_file, index=False)

    # Saving the articles as shards of hdf5 files
    print('Saving articles...')
    os.makedirs(output_directory, exist_ok=True)
    for i, start in tqdm(
            enumerate(range(0, len(filtered_data), items_per_shard))):
        file_name = os.path.join(output_directory, f'shard_{i:04d}.h5')
        end = start + items_per_shard
        save_articles_to_hdf5(filtered_data[start:end],
                              file_name,
                              disable_tqdm=True)
