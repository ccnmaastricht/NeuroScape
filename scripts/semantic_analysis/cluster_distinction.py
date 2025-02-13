import os
import time
import openai
import numpy as np
import pandas as pd
from glob import glob
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from src.utils.parsing import parse_directories
from src.utils.hypersphere import get_centroids
from src.utils.load_and_save import load_embedding_shards, align_to_df
from src.utils.semantic import load_configurations, safe_dictionary_extraction, get_abstract_strings

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

BASEPATH = os.environ['BASEPATH']

# Deine prompt temlates
DEFINITION_PROMPT = PromptTemplate(
    input_variables=["cluster_a_abstracts", "cluster_b_abstracts"],
    template="""
You are provided with two sets of neuroscientific abstracts from Cluster A and Cluster B, which are similar in nature. Analyze the abstracts from both clusters and identify the most distinguishing features that separate Cluster A from Cluster B with high accuracy and conciseness.

**Instructions:**
- Return your findings **only** in JSON format.
- The JSON should contain a single field named `"Distinguishing Features"`.
- Do **not** include any other text, explanations, or comments.
- Do **not** include more than three features.
- Always contrast the features of Cluster A with Cluster B.
- Keep the feature descriptions short (1 or 2 sentences) and simple. Non-experts must be able to understand.

**Example Output:**
```json
{{
  "Distinguishing Features": "Feature 1 description; Feature 2 description; Feature 3 description"
}}


**Cluster A Abstracts:**
{cluster_a_abstracts}

**Cluster B Abstracts:**
{cluster_b_abstracts}
""")

if __name__ == '__main__':
    configurations = load_configurations()
    required_fields = configurations['distinction']['required_fields']
    directories = parse_directories()

    checkpoints_folder = os.path.join(BASEPATH,
                                      directories['internal']['checkpoints'])

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    distinction_chain = DEFINITION_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV files
    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'],
        'Neuroscience')
    article_csv_file = 'articles_merged_cleaned_filtered_clustered.csv'
    cluster_csv_file = 'clusters_defined.csv'
    article_df = pd.read_csv(os.path.join(csv_directory, article_csv_file))

    input_file = os.path.join(csv_directory, cluster_csv_file)
    cluster_definitions_df = pd.read_csv(input_file)
    output_file = input_file.replace('.csv', '_distinguished.csv')
    checkpoint_file = os.path.join(checkpoints_folder,
                                   'cluster_distinction_checkpoint.csv')

    if os.path.exists(checkpoint_file):
        cluster_distinctions_df = pd.read_csv(checkpoint_file)
        cluster_distinctions = cluster_distinctions_df.to_dict('records')
        processed_clusters = set(
            cluster_distinctions_df['Cluster ID'].values.tolist())
    else:
        cluster_distinctions = []
        processed_clusters = set()

    # Load the embeddings and PMIDs
    shard_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['neuro'])
    files = glob(os.path.join(shard_directory, '*.h5'))
    embeddings, pmids = load_embedding_shards(files)

    # Align the embeddings and PMIDs to the DataFrame
    aligned_embeddings, aligned_pmids = align_to_df(embeddings, pmids,
                                                    article_df)

    # Free memory
    del embeddings
    del pmids

    # Get the centroids of the clusters
    centroids = get_centroids(aligned_embeddings,
                              article_df['Cluster ID'].values)
    centroid_similarities = centroids.dot(centroids.T) - np.eye(
        centroids.shape[0])

    # Get cluster labels
    labels = article_df['Cluster ID'].values

    # Get the abstracts
    abstracts = article_df['Abstract'].values

    for label, centroid in enumerate(centroids):

        if label in processed_clusters:
            continue

        cluster_embeddings = aligned_embeddings[labels == label]
        cluster_abstracts = abstracts[labels == label]

        similar_cluster_label = cluster_definitions_df.loc[
            label, 'Most Similar Cluster']

        similar_cluster_embeddings = aligned_embeddings[labels ==
                                                        similar_cluster_label]
        similar_cluster_abstracts = abstracts[labels == similar_cluster_label]
        similar_cluster_centroid = centroids[similar_cluster_label]

        number_of_abstracts_cluster = min(
            configurations['distinction']['max_abstract_number'] // 2,
            len(cluster_abstracts))

        number_of_abstracts_similar_cluster = min(
            configurations['distinction']['max_abstract_number'] // 2,
            len(similar_cluster_abstracts))

        cluster_abstracts = get_abstract_strings(similar_cluster_centroid,
                                                 cluster_embeddings,
                                                 cluster_abstracts,
                                                 number_of_abstracts_cluster)

        similar_cluster_abstracts = get_abstract_strings(
            centroid, similar_cluster_embeddings, similar_cluster_abstracts,
            number_of_abstracts_similar_cluster)

        chain_input = {
            "cluster_a_abstracts": cluster_abstracts,
            "cluster_b_abstracts": similar_cluster_abstracts
        }

        cluster_distinction = safe_dictionary_extraction(
            required_fields, chain_input, distinction_chain,
            configurations['llm']['retries'], configurations['llm']['delay'])

        cluster_distinction['Cluster ID'] = label
        cluster_distinction['Distinguishing Features'] = cluster_distinction[
            'Distinguishing Features'].replace(', ', '; ').replace(
                'A', f'{label}').replace('B', f'{similar_cluster_label}')
        cluster_distinctions.append(cluster_distinction)
        cluster_distinctions_df = pd.DataFrame(cluster_distinctions)

        cluster_distinctions_df.to_csv(checkpoint_file, index=False)

    # Add the cluster distinctions to the DataFrame as a new column
    cluster_definitions_df[
        'Distinguishing Features'] = cluster_distinctions_df[
            'Distinguishing Features']

    # Save the cluster definitions to a CSV file
    cluster_definitions_df.to_csv(os.path.join(csv_directory, output_file),
                                  index=False)
