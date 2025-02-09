import os
import time

import pandas as pd
from tqdm import tqdm

from src.utils.initial_embedding import *
from src.classes.data_types import Embeddings
from src.utils.parsing import parse_directories, parse_discipline
from src.utils.load_and_save import determine_output_filename
from src.utils.checkpoints import save_processed_articles, load_processed_articles

from langchain_voyageai.embeddings import VoyageAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]

if __name__ == "__main__":

    directories = parse_directories()

    embedding_parameters = load_configurations()
    model, sleep_time, batch_size, items_per_shard = unpack_embedding_parameters(
        embedding_parameters)

    checkpoints_folder = os.path.join(BASEPATH,
                                      directories['internal']['checkpoints'])

    discipline = parse_discipline()
    df_dir = os.path.join(BASEPATH,
                          directories['internal']['intermediate']['csv'],
                          discipline)
    embedding_dir = os.path.join(
        BASEPATH, directories['internal']['intermediate']['embeddings'],
        discipline)

    file = os.path.join(df_dir, 'articles_merged_cleaned.csv')
    df = pd.read_csv(file)

    embedding_model = VoyageAIEmbeddings(model=model,
                                         batch_size=batch_size,
                                         voyage_api_key=VOYAGE_API_KEY)

    embedded_articles_file = os.path.join(checkpoints_folder,
                                          'embedded_articles.json')
    embedded_articles = load_processed_articles(embedded_articles_file)

    # remove already embedded articles from df
    df = df[~df['Pmid'].isin(embedded_articles)]

    # check if directory exists
    os.makedirs(embedding_dir, exist_ok=True)

    output_file, shard_id = determine_output_filename(embedding_dir, 'pkl')

    for start in tqdm(range(0, len(df), items_per_shard)):
        end = start + items_per_shard

        abstract_embeddings = Embeddings(pmids=[], embeddings=[])

        selection = df.iloc[start:end]

        abstracts = selection['Abstract'].tolist()
        embedded_abstracts = embedding_model.embed_documents(abstracts)

        abstract_embeddings.pmids = selection['Pmid'].tolist()
        abstract_embeddings.embeddings = embedded_abstracts

        save_embeddings(abstract_embeddings, output_file)

        embedded_articles.update(abstract_embeddings.pmids)

        save_processed_articles(embedded_articles_file, embedded_articles)
        shard_id = shard_id + 1
        output_file = os.path.join(embedding_dir, f'shard_{shard_id:04d}.pkl')
        time.sleep(sleep_time)
