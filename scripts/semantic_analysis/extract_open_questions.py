import os
import re
import time
import openai
import pandas as pd
from glob import glob
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pdfminer.high_level import extract_text
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from scripts.analysis.cluster_definition import load_configurations
from utils.hypersphere import get_centroids, align_to_df
from src.utils.load_and_save import load_embedding_shards

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

BASEPATH = os.environ['BASEPATH']


def process_dictionary(dictionary):
    """
    Process the dictionary extracted from the LLM output.

    Parameters:
    - dictionary: dict, the LLM output dictionary.

    Returns:
    - dictionary: dict, the extracted dictionary.
    """

    try:
        # Verify that the required field is present
        if 'Open Questions' in dictionary:
            dictionary['Open Questions'] = dictionary[
                'Open Questions'].replace(',', ';')
            return dictionary
        else:
            raise ValueError(
                "Missing required 'Open Questions' field in the LLM output")

    except Exception:
        return None


def safe_dictionary_extraction(cluster_title, cluster_definition, reviews,
                               chain, retries, delay):
    """
    Safely extract the dictionary from the LLM output.

    Parameters:
    - 
    - chain: LLMChain, the LLM chain to invoke.
    - retries: int, the number of retries allowed.
    - delay: int, the delay between retries.

    Returns:
    - dictionary: dict, the extracted dictionary.
    """

    attempt = 0
    chain_input = {
        "cluster_title": cluster_title,
        "cluster_definition": cluster_definition,
        "reviews": reviews
    }
    while attempt < retries:
        dictionary = chain.invoke(chain_input)
        open_questions = process_dictionary(dictionary)
        time.sleep(delay)
        if open_questions:
            return open_questions
        else:
            attempt += 1

    return None


def strip_document(document):
    """
    Strip the document of any unwanted sections.

    Parameters:
    - document: str, the document to strip.

    Returns:
    - document: str, the stripped document.
    """

    unnecessary_sections = [
        'references', 'acknowledgements', 'author contributions', 'funding',
        'funding sources', 'conflict of interest',
        'conflict of interest statement'
    ]

    # Convert document to lowercase for case-insensitive matching
    document_lower = document.lower()

    # Use regular expressions to strip the document starting from 'introduction' or 'main'
    introduction_match = re.search(r'\bintroduction\b', document_lower,
                                   re.IGNORECASE)
    main_match = re.search(r'\bmain\b', document_lower, re.IGNORECASE)

    if introduction_match:
        document = document[introduction_match.end():]
    elif main_match:
        document = document[main_match.end():]

    # Use regular expressions to remove unnecessary sections
    for section in unnecessary_sections:
        pattern = re.compile(r'\b' + re.escape(section) + r'\b', re.IGNORECASE)
        match = pattern.search(document)
        if match:
            document = document[:match.start()]

    return document


def get_review_text(files):
    """
    Get the abstract strings that are most similar to the centroid.

    Parameters:
    - files: list, the list of pdf files to extract text from.

    Returns:
    - review_text: str, the review text.
    """

    review_text = ''
    for file in files:
        document = extract_text(file)
        document = strip_document(document)
        review_text = f"""{review_text}\n\n{document}"""

    return review_text


# Deine prompt temlates
QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["cluster_title", "cluster_definition", "reviews"],
    template="""

    You are a domain expert in neuroscience. You have the combined text of several recent neuroscience review articles that all pertain to the same cluster entitled "{cluster_title}". 
    {cluster_definition}
    Your task is to synthesize and prioritize the major open questions explicitly or implicitly mentioned in these articles.

    Below are the texts (or substantial excerpts) of several review articles that cover this cluster. Each article might mention multiple open questions, challenges, or research gaps. Please read them carefully and produce a consolidated list of the major open questions in this subdomain.

**Instructions:**
- Return your findings **only** in JSON format (valid JSON, no extra text).
- The JSON should contain a single field named `"Open Questions"`.
- Do **not** include any other text, explanations, or comments.
- Do **not** include more than five Open Questions.
- No lists: Provide the Open Questions as a single string separated by semicolons, not as a list.
- Highlight overlap: Prioritize questions that appear in multiple reviews.
- Avoid fabrications: Only list questions that are actually stated or strongly implied by the articles.
- Condense duplication: If multiple reviews mention the same open question, merge them into a single entry.
- Preserve specificity: Include enough detail to capture the essence of each open question.
- Be concise: Keep each open question short (2 or 3 sentences) and simple. Non-experts must be able to understand.

**Example Output:**
```json
{{
  "Open Questions": "Open question 1; Open question 2; Open question 3"
}}


**Reviews:**
{reviews}
""")

if __name__ == '__main__':
    configurations = load_configurations()

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    questions_chain = QUESTIONS_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV files
    pdf_directory = os.path.join(BASEPATH,
                                 configurations['data']['pdf_directory'])
    csv_directory = os.path.join(BASEPATH,
                                 configurations['data']['csv_directory'])
    cluster_csv_file = 'neuroscience_articles_1999-2023_cluster_descriptions_density.csv'

    output_path = os.path.join(csv_directory, cluster_csv_file)
    cluster_definitions_df = pd.read_csv(output_path)
    interim_path = output_path.replace('.csv', '_interim.csv')
    output_path = output_path.replace('.csv', '_questions.csv')

    if os.path.exists(interim_path):
        cluster_questions_df = pd.read_csv(interim_path)
        cluster_questions = cluster_questions_df.to_dict('records')
        processed_clusters = set(
            cluster_questions_df['Cluster ID'].values.tolist())

        print(f'{len(processed_clusters)} clusters have been processed.')
    else:
        cluster_questions = []
        processed_clusters = set()

    unique_clusters = sorted(
        cluster_definitions_df['Cluster ID'].values.tolist())

    for cluster in tqdm(unique_clusters):

        if cluster in processed_clusters:
            continue

        cluster_pdf_directory = os.path.join(
            pdf_directory, f'cluster_{str(cluster).zfill(3)}')

        pdf_files = glob(os.path.join(cluster_pdf_directory, '*.pdf'))
        cluster_title = cluster_definitions_df.loc[cluster, 'Title']
        cluster_definition = cluster_definitions_df.loc[cluster, 'Description']

        reviews = get_review_text(pdf_files)

        open_questions = safe_dictionary_extraction(
            cluster_title, cluster_definition, reviews, questions_chain,
            configurations['llm']['retries'], configurations['llm']['delay'])

        open_questions['Cluster ID'] = cluster
        cluster_questions.append(open_questions)
        cluster_questions_df = pd.DataFrame(cluster_questions)
        cluster_questions_df.to_csv(interim_path, index=False)

    # Add the open questions as a bew column to the cluster definitions
    cluster_definitions_df['Open Questions'] = cluster_questions_df[
        'Open Questions']

    # Save the cluster definitions to a CSV file
    cluster_definitions_df.to_csv(output_path, index=False)
