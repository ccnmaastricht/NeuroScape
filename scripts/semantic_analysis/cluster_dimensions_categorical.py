import os
import time
import openai
import tomllib
import pandas as pd

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
    with open('config/analysis/cluster_dim_categorical.toml', 'rb') as f:
        configurations = tomllib.load(f)

    return configurations


def process_dictionary(dictionary, required_keys):
    """
    Process the dictionary extracted from the LLM output.

    Parameters:
    - dictionary: dict, the LLM output dictionary.
    - required_keys: list, the required keys.
    - allowed_values: list, the allowed values for the required keys.

    Returns:
    - new_dict: dict, the extracted dictionary.
    """

    allowed_values = ['yes', 'no']
    conversion = {"yes": True, "no": False}

    try:
        # Verify that the required fields are present
        if not all(key in dictionary for key in required_keys):
            raise ValueError("Missing required fields in the LLM output")

        new_dict = {
            key: conversion[dictionary[key]]
            for key in required_keys if dictionary[key] in allowed_values
        }
        if len(new_dict) != len(required_keys):
            raise ValueError("Invalid values in the LLM output")

        return new_dict

    except Exception:
        return None


def safe_dictionary_extraction(dimension, categories, dimensions_text, chain,
                               retries, delay):
    """
    Safely extract the dictionary from the LLM output.

    Parameters:
    - dimension: str, the dimension to extract.
    - categories: list, the categories.
    - dimensions_text: str, the dimensions text.
    - chain: LLMChain, the LLM chain to invoke.
    - retries: int, the number of retries allowed.
    - delay: int, the delay between retries.

    Returns:
    - dictionary: dict, the extracted dictionary.
    """
    categories_text = "\n".join(categories)
    chain_input = {
        "dimension": dimension,
        "categories": categories_text,
        "analysis": dimensions_text
    }
    attempt = 0
    while attempt < retries:
        dictionary = chain.invoke(chain_input)
        cluster_dimensions = process_dictionary(dictionary,
                                                required_keys=categories)
        time.sleep(delay)
        if cluster_dimensions:
            return cluster_dimensions
        else:
            attempt += 1

    return None


def get_dimension_df(csv_path, dim_id, dimension_df, dimension, categories,
                     chain, retries, delay):
    filename = f"dimension_{dim_id}_{dimension.lower().replace(' ','_')}.csv"

    if os.path.exists(os.path.join(csv_path, filename)):
        print('Loading from CSV')
        return pd.read_csv(os.path.join(csv_path, filename))

    print('Processing')
    dimension_dict = []
    for cluster in tqdm(dimension_df.index):
        analysis = dimension_df.loc[cluster, "Dimensions"]
        cluster_dict = safe_dictionary_extraction(dimension, categories,
                                                  analysis, chain, retries,
                                                  delay)
        cluster_dict['Cluster ID'] = cluster
        dimension_dict.append(cluster_dict)

    dimension_df = pd.DataFrame(dimension_dict)
    dimension_df.to_csv(os.path.join(csv_path, filename), index=False)
    return pd.DataFrame(dimension_dict)


DIMENSIONS_PROMPT = PromptTemplate(
    input_variables=["dimension", "categories", "analysis"],
    template="""
            You are an expert in neuroscience. 
            You are provided with an analysis of research within a neuroscientific cluster along the following 9 dimensions:

            1. Appliedness: The extent to which the research is basic science (fundamental) or applied in one of several ways.
            2. Modality: The sensory and/or motor modality under investigation.
            3. Spatiotemporal Scale: The spatial and temporal scale of the research. Can vary from microscale to macroscale for both space and time.
            4. Cognitive Complexity: The level of cognitive complexity under investigation from low level (e.g., sensory processing, motor control) to high level (e.g., language, decision making, social cognition).
            5. Species: The species under investigation.
            6. Theory Engagement: The extent to which the research is theory-driven (hypothesis testing) or data-driven (exploratory, descriptive).
            7. Theory Scope: The scope of the theory under investigation. Overarching Framework: In neuroscience, an overarching framework is a broad theoretical approach that aims to explain fundamental principles of brain function across multiple cognitive and neural domains.
                            Domain Framework: In neuroscience, a domain framework is an integrative theory focusing on a specific subfield, offering cohesive principles for that area.
                            Disease-specific Framework: In neuroscience, a disease-specific framework is a theory that details the neural causes, mechanisms, and manifestations of a particular neurological or psychiatric condition.
                            Micro Theory: In neuroscience, a micro theory is a narrowly scoped, mechanistic account or model that explains one specific process or phenomenon in the brain.
            8. Methodological Approach: The methodological approach used in the research. Experimental: Studies in which researchers deliberately manipulate one or more variables under controlled conditions to test for causal effects.
                                        Observational (Correlational / Descriptive): Studies that measure variables in naturally occurring settings without introducing any active intervention, focusing on describing or correlating observed phenomena.
                                        Computational / coding: Studies that construct or test mathematical, algorithmic, or simulation-based models to predict, explain, or interpret empirical data or biological processes.
                                        Theoretical / Conceptual: Work that develops, refines, or critiques conceptual frameworks and theories without generating new empirical data or running computational simulations.
                                        Meta-Analytic / Systematic Review: Research that synthesizes and reanalyzes existing primary studies, systematically aggregating findings using quantitative (meta-analysis) or rigorous protocol-based (systematic review) methods.
            9. Interdisciplinarity: The extent to which the research is interdisciplinary, combining methods and concepts from multiple fields. From low (confined to a single discipline) to very high (incorporating multiple disciplines in a transcdisciplinary manner).
                                    Multidisciplinary: Multiple disciplines study the same problem in parallel, each applying its own methods and perspectives but with little cross-integration.
                                    Interdisciplinary: Researchers from different disciplines integrate theories, methods, or data to create shared frameworks or solutions that transcend any single field.
                                    Transdisciplinary: Collaboration goes beyond standard academic boundaries, involving non-academic stakeholders or merging disciplines so completely that new fields or holistic approaches emerge.

            Your task is to focus solely on the dimension of {dimension} and provide a binary indication ("yes" or "no") of whether the research within the cluster falls within the specified categories.
            Here are the categories for this dimension:
            {categories}

            **Output Format:**

            Please present your findings in **JSON format** with the following structure:
            {{
                "Category 1": "yes" / "no",
                "Category 2": "yes" / "no",
                "Category 3": "yes" / "no",
                ...
                category n: "yes" / "no"
            }}

            ```

            **Instructions:**
            - **Accuracy is crucial**: Ensure all information is directly supported by the provided analysis. Do not include information not present in the analysis or make external assumptions.

            - **Consisteny**: Ensure that the evaluation of the dimension does not contradict the provided analysis (including other dimensions).
            
            - **Focus and Precision**: Only evaluate the dimension of {dimension} and provide a binary response for each category. Do not include any additional information or explanations.

            - **Proper Category Naming**: Ensure that the categories are named correctly and accurately reflect the content of the analysis. ONLY use the provided categories and replace Category 1, Category 2, etc. with the actual category names.

            - **Binary Response**: Ensure that the response for each category is binary (yes or no) and does not include any other text or explanations.


            **Here is the analysis of the research within the cluster along the dimension of {dimension}:**
            {analysis}""",
)

if __name__ == '__main__':
    configurations = load_configurations()

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    dimensions_chain = DIMENSIONS_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV file
    csv_directory = os.path.join(BASEPATH,
                                 configurations['data']['csv_directory'])
    cluster_csv_file = configurations['data']['input_file_name']
    cluster_df = pd.read_csv(os.path.join(csv_directory, cluster_csv_file))
    narrative_dimension_df = cluster_df[["Cluster ID", "Dimensions"]]

    # Define the output path
    dimension_csv_file = configurations['data']['output_file_name']
    dimension_categories = {
        'Appliedness': [
            'Fundamental', 'Translational', 'Clinical', 'Method Development',
            'Technological Exploitation'
        ],
        'Modality': [
            'Auditory', 'Visual', 'Olfactory', 'Gustatory', 'Somatosensory',
            'Multimodal', 'Visuomotor', 'Sensorimotor', 'Motor', 'Emotional',
            'Behavioral', 'Cognitive'
        ],
        'Spatial Scale': [
            'Molecular', 'Cellular', 'Circuit', 'Region', 'Systems',
            'Whole-brain'
        ],
        'Temporal Scale': [
            'Microsecond', 'Millisecond', 'Second', 'Minute', 'Hour', 'Day',
            'Week', 'Month', 'Year', 'Lifetime'
        ],
        'Cognitive Complexity':
        ['Low-level Sensory', 'Low-level Motor', 'Mid-level', 'High-level'],
        'Species': [
            'Human', 'Non-human primates', 'Rodents', 'Mammals', 'Birds',
            'Fish', 'Amphibians', 'Invertebrates', 'Cell cultures', 'Other'
        ],
        'Theory Engagement': ['Data-driven', 'Hypothesis-driven'],
        'Theory Scope': [
            'Overarching Framework', 'Domain Framework',
            'Disease-specific Framework', 'Micro Theory'
        ],
        'Methodological Approach': [
            'Experimental', 'Observational', 'Computational', 'Theoretical',
            'Meta-analytic'
        ],
        'Interdisciplinarity': ['Low', 'Medium', 'High', 'Very High']
    }

    for dimension_id, (dimension,
                       categories) in enumerate(dimension_categories.items(),
                                                start=1):

        dimension_df = get_dimension_df(csv_directory, dimension_id,
                                        narrative_dimension_df, dimension,
                                        categories, dimensions_chain,
                                        configurations['llm']['retries'],
                                        configurations['llm']['delay'])

        data = {
            f"{dimension}_{category}": dimension_df[category].values
            for category in categories
        }

        if dimension_id == 1:
            dim_cat_df = pd.DataFrame(data)
        else:
            dim_cat_df = pd.concat([dim_cat_df, pd.DataFrame(data)], axis=1)

    # Save the cluster definitions to a CSV file
    dim_cat_df.to_csv(os.path.join(csv_directory, dimension_csv_file),
                      index=False)
