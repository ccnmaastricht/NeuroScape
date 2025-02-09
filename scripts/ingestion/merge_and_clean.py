import os
import glob
from src.utils.cleaning import *
from src.utils.parsing import parse_directories, parse_discipline

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']

if __name__ == '__main__':
    directories = parse_directories()
    config = load_configurations()

    cutoffs = (config['word_limit']['lower'], config['word_limit']['upper'],
               config['year_cutoff'])
    discipline = parse_discipline()
    raw_directory = os.path.join(BASEPATH, directories['internal']['raw'],
                                 discipline)
    cleaned_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['csv'], discipline)
    raw_files = glob.glob(os.path.join(raw_directory, '*.csv'))

    print(f'Merging {len(raw_files)} files.')
    df = concatenate_files(raw_files)
    df['Pmid'] = df['Pmid'].astype(int)

    print(f'Cleaning the merged dataframe with {len(df)} articles.')
    df = clean_dataframe(df, cutoffs)

    print(f'Number of articles after cleaning: {len(df)}')
    print('Sorting the clean dataframe.')
    df = sort_dataframe(df)

    print(f'Saving the clean dataframe to {cleaned_directory}.')
    os.makedirs(cleaned_directory, exist_ok=True)
    df.to_csv(os.path.join(cleaned_directory, 'articles_merged_cleaned.csv'),
              index=False)
