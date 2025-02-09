import os
import time
import openai
import tomllib
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

# Define prompt temlates
DEFINITION_PROMPT = PromptTemplate(
    input_variables=["abstracts"],
    template="""
            You are provided with a list of scientific abstracts that belong to a specific research cluster. Your task is to:

1. **Identify the most frequent and relevant keywords and phrases** used in the abstracts within this cluster. The keywords should accurately characterize the research within the cluster and avoid wrong assertions.

2. **Provide a descriptive title** for the cluster that encapsulates the main themes and methodologies.

3. **Write a brief summary** (1-2 sentences) that describes the main themes and methodologies of the cluster.

4. **Determine the main focus** of the cluster, whether it is on the themes or methodologies. A methodological focus is when a cluster focuses on method development or consistently applies a specific methodology. A thematic focus is when a cluster consistently studies a specific phenomenon.

**Output Format:**

Please present your findings in **JSON format** with the following structure:

  "Keywords": ["keyword1", "keyword2", "keyword3", ...],
  "Title": "Descriptive Cluster Title",
  "Description": "Brief summary of the cluster's main themes and methodologies."
  "Focus": "The main focus of the cluster is either on the themes or methodologies."
```

**Instructions:**

- **Accuracy is crucial**: Ensure all information is directly supported by the provided abstracts. Do not include information not present in the abstracts or make external assumptions.

- **Clarity and Precision**: The keywords, title, and description should be clear and accurately reflect the content of the abstracts.

- **Conciseness**: Do not include any additional text or explanations beyond the specified JSON output. Do not generate more output than necessary.

**Here are the abstracts:**
{abstracts}""",
)

if __name__ == '__main__':
    configurations = load_configurations()
    directories = parse_directories()

    required_fields = configurations['definition']['required_fields']

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    definition_chain = DEFINITION_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV file
    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'],
        'Neuroscience')
    article_csv_file = 'articles_merged_cleaned_filtered_clustered.csv'
    article_df = pd.read_csv(os.path.join(csv_directory, article_csv_file))

    # Define the output path
    cluster_csv_file = 'clusters_defined.csv'
    output_path = os.path.join(csv_directory, cluster_csv_file)

    if os.path.exists(output_path):
        cluster_definitions_df = pd.read_csv(output_path)
        cluster_definitions = cluster_definitions_df.to_dict('records')
        processed_clusters = set(
            cluster_definitions_df['Cluster ID'].values.tolist())
    else:
        cluster_definitions = []
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

    # Get the publication years of the articles
    article_years = article_df['Year'].values

    # Get the citation rates of the articles
    citation_rates = article_df['Citation Rate'].values

    # Get the type of the articles
    article_types = article_df['Type'].values

    for label, centroid in enumerate(centroids):

        if label in processed_clusters:
            continue

        cluster_embeddings = aligned_embeddings[labels == label]
        cluster_abstracts = abstracts[labels == label]

        number_of_abstracts = min(
            configurations['definition']['max_abstract_number'],
            len(cluster_abstracts))

        cluster_abstracts = get_abstract_strings(centroid, cluster_embeddings,
                                                 cluster_abstracts,
                                                 number_of_abstracts)

        chain_input = {"abstracts": cluster_abstracts}
        cluster_definition = safe_dictionary_extraction(
            required_fields, chain_input, definition_chain,
            configurations['llm']['retries'], configurations['llm']['delay'])

        # Join the keywords into a single string
        cluster_definition['Keywords'] = '; '.join(
            cluster_definition['Keywords'])

        cluster_definition['Cluster ID'] = label
        cluster_definition['Size'] = sum(labels == label)
        cluster_definition['Year First Article'] = article_years[labels ==
                                                                 label].min()
        cluster_definition['MCR Research'] = np.median(
            citation_rates[(labels == label) & (article_types == 'Research')])
        cluster_definition['MCR Review'] = np.median(
            citation_rates[(labels == label) & (article_types == 'Review')])

        most_similar = np.argmax(centroid_similarities[label])
        cluster_definition['Most Similar Cluster'] = most_similar
        cluster_definition['Similarity'] = centroid_similarities[label,
                                                                 most_similar]

        cluster_definitions.append(cluster_definition)

    # Convert the cluster definitions to a DataFrame
    cluster_definitions_df = pd.DataFrame(cluster_definitions)

    # Reorder the columns
    cluster_definitions_df = cluster_definitions_df[[
        'Cluster ID', 'Title', 'Size', 'Year First Article', 'MCR Research',
        'MCR Review', 'Keywords', 'Description', 'Focus',
        'Most Similar Cluster', 'Similarity'
    ]]

    # Save the cluster definitions to a CSV file
    cluster_definitions_df.to_csv(output_path, index=False)
