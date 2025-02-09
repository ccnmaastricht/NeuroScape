import os
import time
import openai
import tomllib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

BASEPATH = os.environ['BASEPATH']


# Define utility functions
def load_configurations():
    """
    Load the configuration from the config file.
    
    Returns:
    - configurations: dict
    """
    with open('config/analysis/cluster_definition.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def process_dictionary(dictionary):
    """
    Process the dictionary extracted from the LLM output.

    Parameters:
    - dictionary: dict, the LLM output dictionary.

    Returns:
    - new_dict: dict, the extracted dictionary.
    """

    try:
        # Verify that the required fields are present
        if not all(key in dictionary for key in [
                'Emerging Themes', 'Emerging Methodological Approaches',
                'Declining Themes', 'Declining Methodological Approaches',
                'Nature Research'
        ]):
            raise ValueError("Missing required fields in the LLM output")

        # Join the keys as a single string with each key as a title and a line break between them
        new_dict = {
            'Trends':
            '\n'.join([f'{key}: {value}' for key, value in dictionary.items()])
        }

        return new_dict

    except Exception:
        return None


def safe_dictionary_extraction(cluster_title, year, old_abstracts,
                               recent_abstracts, chain, retries, delay):
    """
    Safely extract the dictionary from the LLM output.

    Parameters:
    - cluster_title: String, the cluster
    - year: int, the year to compare.
    - old_abstracts: String, the old abstracts.
    - recent_abstracts: String, the recent abstracts.
    - chain: LLMChain, the LLM chain to invoke.
    - retries: int, the number of retries allowed.
    - delay: int, the delay between retries.

    Returns:
    - dictionary: dict, the extracted dictionary.
    """
    chain_input = {
        "title": cluster_title,
        "year": year,
        "old_abstracts": old_abstracts,
        "recent_abstracts": recent_abstracts
    }
    attempt = 0
    while attempt < retries:
        dictionary = chain.invoke(chain_input)
        cluster_trends = process_dictionary(dictionary)
        time.sleep(delay)
        if cluster_trends:
            return cluster_trends
        else:
            attempt += 1

    return None


# Deine prompt temlates
TRENDS_PROMPT = PromptTemplate(
    input_variables=["title", "year", "old_abstracts", "recent_abstracts"],
    template="""
            You are an expert in neuroscience and scientific text analysis. 
            You are provided with a set of older neuroscientific abstracts and recent neuroscientific abstracts that belong to the cluster '{title}'.
            
            Your task is to compare older abstracts (pre-{year}) to recent abstracts (post-{year}), focusing on:

            Emerging thematic trends: New or increasingly emphasized topics (as well as theories) that appear predominantly in the recent set.
            Emerging methodological trends: New tools, techniques, or approaches introduced or significantly gaining traction in the recent set.
            Declining themes: Topics or themes that were prominent in the older set but are less emphasized or absent in the recent set.
            Declining methodological approaches: Tools, techniques, or approaches that were commonly used in the older set but are now less represented or obsolete.
            Provide your analysis in a structured JSON format.

            **Output Format:**

            Please present your findings in **JSON format** with the following structure:
            {{
                "Emerging Themes": "Semicolon separated list of emerging thematic trends with brief descriptions",
                "Emerging Methodological Approaches": "Semicolon separated list of emerging methods or techniques with brief descriptions",
                "Declining Themes": "Semicolon separated list of themes with brief descriptions that are less emphasized or absent",
                "Declining Methodological Approaches": "Semicolon separated list of older methods or techniques with brief descriptions that are now less used"
            }}

            **Instructions:**
            - **No colons**: Do not use colons in the JSON output.
            - **Accuracy is crucial**: Ensure all information is directly supported by the provided abstracts. Do not include information not present in the abstracts or make external assumptions.
            - **Clarity and Precision**: Assessments and descriptions should be clear and accurately reflect the content of the abstracts.
            - **Conciseness**: Do not include any additional text or explanations beyond the specified JSON output. Do not generate more output than necessary.

            **Here are the older abstracts**:
            {old_abstracts}

            **Here are the recent abstracts**:
            {recent_abstracts}
            """,
)

if __name__ == '__main__':
    configurations = load_configurations()
    num_abstracts_per_set = configurations['sampling'][
        'max_abstract_number'] // 2

    recent_cutoff = configurations['sampling']['recent_cutoff']
    older_cutoff = configurations['sampling']['older_cutoff']

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    trends_chain = TRENDS_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV file
    csv_directory = os.path.join(BASEPATH,
                                 configurations['data']['csv_directory'])
    article_csv_file = configurations['data']['input_file_name']
    article_df = pd.read_csv(os.path.join(csv_directory, article_csv_file))
    article_df = article_df[article_df['Type'] == 'Research']

    # Define the output path
    cluster_csv_file = configurations['data']['output_file_name']
    cluster_df = pd.read_csv(os.path.join(csv_directory, cluster_csv_file))
    output_path = os.path.join(csv_directory, cluster_csv_file)
    interim_path = output_path.replace('.csv', '_interim.csv')
    output_path = output_path.replace('.csv', '_trends.csv')

    if os.path.exists(interim_path):
        cluster_trends_df = pd.read_csv(interim_path)
        cluster_trends = cluster_trends_df.to_dict('records')
        processed_clusters = set(
            cluster_trends_df['Cluster ID'].values.tolist())

        print(f'{len(processed_clusters)} clusters have been processed.')
    else:
        cluster_trends = []
        processed_clusters = set()

    # Keep only the research articles
    research_articles = article_df[article_df['Type'] == 'Research']

    # Get cluster labels
    labels = research_articles['Cluster'].values

    for label in tqdm(np.unique(labels)):

        if label in processed_clusters:
            continue

        older_abstracts = research_articles[
            (research_articles['Cluster'] == label)
            & (research_articles['Year'] >= older_cutoff) &
            (research_articles['Year'] < recent_cutoff)]['Abstract'].values
        number_of_abstracts = min(num_abstracts_per_set, len(older_abstracts))
        random_indices = np.random.choice(len(older_abstracts),
                                          number_of_abstracts,
                                          replace=False)
        older_abstracts = '\n\n'.join(older_abstracts[random_indices])

        recent_df = research_articles[
            (research_articles['Cluster'] == label)
            & (research_articles['Year'] >= recent_cutoff)]
        # sort by citation rate
        recent_df = recent_df.sort_values('Citation Rate', ascending=False)
        recent_abstracts = recent_df['Abstract'].values[:num_abstracts_per_set]

        number_of_abstracts = min(num_abstracts_per_set, len(recent_abstracts))
        recent_abstracts = '\n\n'.join(recent_abstracts[:number_of_abstracts])

        cluster_title = cluster_df.loc[label, 'Title']

        cluster_trends_dict = safe_dictionary_extraction(
            cluster_title, recent_cutoff, older_abstracts, recent_abstracts,
            trends_chain, configurations['llm']['retries'],
            configurations['llm']['delay'])

        cluster_trends_dict['Cluster ID'] = label
        cluster_trends.append(cluster_trends_dict)
        processed_clusters.add(label)

        cluster_trends_df = pd.DataFrame(cluster_trends)
        cluster_trends_df.to_csv(interim_path, index=False)

    # Convert the cluster definitions to a DataFrame
    cluster_trends_df = pd.DataFrame(cluster_trends)

    # Merge the cluster definitions with the existing cluster definitions
    cluster_trends_df = cluster_df.merge(cluster_trends_df, on='Cluster ID')

    # Save the cluster definitions to a CSV file
    cluster_trends_df.to_csv(output_path, index=False)
