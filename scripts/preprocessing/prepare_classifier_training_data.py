import os
import glob
import time
import pandas as pd

from sklearn.model_selection import train_test_split

from src.utils.parsing import parse_directories
from src.classes.data_types import EmbeddingsWithLabels
from src.utils.classifier import load_configurations, save_and_create_dataset, \
    load_shard, extract_data, get_disciplines, generate_n_hot_vector, get_unique_disciplines_and_count

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']

seed = int(time.time())


def process_shards(shards,
                   dataframe,
                   unique_disciplines,
                   shard_ids,
                   directories,
                   set_type,
                   threshold,
                   delete_shards=False):
    """
    Process a list of shards and save the data to the corresponding dataset.

    Parameters:
    shards (list): A list of shards.
    dataframe (pd.DataFrame): The dataframe containing the disciplines.
    unique_disciplines (list): A list of unique disciplines.
    shard_ids (tuple): A tuple containing the ids of the shards.
    directories (tuple): A tuple containing the directories where the shards will be saved.
    threshold (int): The threshold of the number of instances in a dataset.
    delete_shards (bool): A flag to delete the shards after processing.

    Returns:
    int: The id of the mono dataset.
    int: The id of the multi dataset.
    """
    mono_data = EmbeddingsWithLabels(pmids=[], embeddings=[], labels=[])
    multi_data = EmbeddingsWithLabels(pmids=[], embeddings=[], labels=[])
    num_mono = 0
    num_multi = 0

    id_mono, id_multi = shard_ids
    monolabel_directory, multilabel_directory = directories

    for shard in shards:
        if num_mono >= threshold:
            mono_data, id_mono = save_and_create_dataset(
                mono_data, id_mono, monolabel_directory, set_type)
            num_mono = 0

        if num_multi >= threshold:
            multi_data, id_multi = save_and_create_dataset(
                multi_data, id_multi, multilabel_directory, set_type)
            num_multi = 0

        abstracts = load_shard(shard)
        if delete_shards:
            os.remove(shard)

        shard_disciplines = [
            get_disciplines(dataframe, pmid) for pmid in abstracts.pmids
        ]
        n_hot_vector, num_hot = generate_n_hot_vector(shard_disciplines,
                                                      unique_disciplines)

        mono_pmids, mono_embeddings, mono_labels = extract_data(
            abstracts, n_hot_vector, num_hot, lambda x: x == 1)
        mono_data.pmids.extend(mono_pmids)
        mono_data.embeddings.extend(mono_embeddings)
        mono_data.labels.extend(mono_labels)
        num_mono += len(mono_pmids)

        multi_pmids, multi_embeddings, multi_labels = extract_data(
            abstracts, n_hot_vector, num_hot, lambda x: x > 1)
        multi_data.pmids.extend(multi_pmids)
        multi_data.embeddings.extend(multi_embeddings)
        multi_data.labels.extend(multi_labels)
        num_multi += len(multi_pmids)

    return id_mono, id_multi


if __name__ == "__main__":

    configurations = load_configurations()

    data_directories = parse_directories()

    dataframe_directory = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['csv'])
    embedding_directory = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['embeddings'])

    other_dataframe_directory = os.path.join(dataframe_directory,
                                             'OtherDisciplines')
    neuro_dataframe_directory = os.path.join(dataframe_directory,
                                             'Neuroscience')

    other_embedding_directory = os.path.join(embedding_directory,
                                             'OtherDisciplines')
    neuro_embedding_directory = os.path.join(embedding_directory,
                                             'Neuroscience')

    multilabel_directory = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['classifier'],
        'Multilabel')
    monolabel_directory = os.path.join(
        BASEPATH, data_directories['internal']['intermediate']['classifier'],
        'Monolabel')
    os.makedirs(os.path.join(multilabel_directory, 'Train'), exist_ok=True)
    os.makedirs(os.path.join(multilabel_directory, 'Val'), exist_ok=True)
    os.makedirs(os.path.join(multilabel_directory, 'Test'), exist_ok=True)
    os.makedirs(os.path.join(monolabel_directory, 'Train'), exist_ok=True)
    os.makedirs(os.path.join(monolabel_directory, 'Val'), exist_ok=True)
    os.makedirs(os.path.join(monolabel_directory, 'Test'), exist_ok=True)

    other_dataframe = pd.read_csv(
        os.path.join(other_dataframe_directory, 'merged.csv'))
    neuro_dataframe = pd.read_csv(
        os.path.join(neuro_dataframe_directory, 'merged.csv'))
    other_shards = glob.glob(os.path.join(other_embedding_directory, '*.pkl'))
    neuro_shards = glob.glob(os.path.join(neuro_embedding_directory, '*.pkl'))

    unique_disciplines, num_classes = get_unique_disciplines_and_count(
        other_dataframe)

    directories = (monolabel_directory, multilabel_directory)
    threshold = configurations['preparation']['item_threshold']

    train_val_shards, test_shards = train_test_split(other_shards,
                                                     test_size=0.1,
                                                     random_state=seed)
    train_shards, val_shards = train_test_split(train_val_shards,
                                                test_size=0.1,
                                                random_state=seed)

    shard_ids = (0, 0)

    print('Processing training shards of other disciplines...')
    other_train_ids = process_shards(train_shards, other_dataframe,
                                     unique_disciplines, shard_ids,
                                     directories, 'Train', threshold)

    print('Processing validation of other disciplines...')
    other_val_ids = process_shards(val_shards, other_dataframe,
                                   unique_disciplines, shard_ids, directories,
                                   'Val', threshold)

    print('Processing test shards of other disciplines...')
    other_test_ids = process_shards(test_shards, other_dataframe,
                                    unique_disciplines, shard_ids, directories,
                                    'Test', threshold)

    used_shards, _ = train_test_split(neuro_shards,
                                      test_size=0.9,
                                      random_state=seed)
    train_val_shards, test_shards = train_test_split(used_shards,
                                                     test_size=0.1,
                                                     random_state=seed)
    train_shards, val_shards = train_test_split(train_val_shards,
                                                test_size=0.1,
                                                random_state=seed)

    print('Processing training shards of neuroscience...')
    neuro_train_ids = process_shards(train_shards,
                                     neuro_dataframe,
                                     unique_disciplines,
                                     other_train_ids,
                                     directories,
                                     'Train',
                                     threshold,
                                     delete_shards=True)

    print('Processing validation shards of neuroscience...')
    neuro_val_ids = process_shards(val_shards,
                                   neuro_dataframe,
                                   unique_disciplines,
                                   other_val_ids,
                                   directories,
                                   'Val',
                                   threshold,
                                   delete_shards=True)

    print('Processing test shards of neuroscience...')
    neuro_test_ids = process_shards(test_shards,
                                    neuro_dataframe,
                                    unique_disciplines,
                                    other_test_ids,
                                    directories,
                                    'Test',
                                    threshold,
                                    delete_shards=True)

    print('Data preparation completed.')
