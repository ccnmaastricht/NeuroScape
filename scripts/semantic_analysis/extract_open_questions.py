import os
import openai
import pandas as pd
from glob import glob
from tqdm import tqdm

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from src.utils.parsing import parse_directories
from src.utils.semantic import load_configurations, safe_dictionary_extraction, get_review_text

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

BASEPATH = os.environ['BASEPATH']

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
    required_fields = configurations['questions']['required_fields']
    directories = parse_directories()

    llm = ChatOpenAI(temperature=configurations['llm']['temperature'],
                     model_name=configurations['llm']['model_name'])
    questions_chain = QUESTIONS_PROMPT | llm | SimpleJsonOutputParser()

    # Load the CSV files internal.reference.pdf
    checkpoint_path = os.path.join(BASEPATH,
                                   directories['internal']['checkpoints'])
    pdf_directory = os.path.join(BASEPATH,
                                 directories['internal']['reference']['pdfs'])
    csv_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'],
        'Neuroscience')
    cluster_csv_file = 'clusters_defined_distinguished.csv'

    output_file = os.path.join(csv_directory, cluster_csv_file)
    cluster_definitions_df = pd.read_csv(output_file)
    checkpoint_file = os.path.join(checkpoint_path, 'questions_checkpoint.csv')
    output_file = output_file.replace('.csv', '_questions.csv')

    if os.path.exists(checkpoint_file):
        cluster_questions_df = pd.read_csv(checkpoint_file)
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

        chain_input = {
            "cluster_title": cluster_title,
            "cluster_definition": cluster_definition,
            "reviews": reviews
        }
        open_questions = safe_dictionary_extraction(
            required_fields, chain_input, questions_chain,
            configurations['llm']['retries'], configurations['llm']['delay'])

        open_questions['Cluster ID'] = cluster
        cluster_questions.append(open_questions)
        cluster_questions_df = pd.DataFrame(cluster_questions)
        cluster_questions_df.to_csv(checkpoint_file, index=False)

    # Add the open questions as a bew column to the cluster definitions
    cluster_definitions_df['Open Questions'] = cluster_questions_df[
        'Open Questions']

    # Save the cluster definitions to a CSV file
    cluster_definitions_df.to_csv(output_file, index=False)
