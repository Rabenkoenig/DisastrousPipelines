import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them on the 'id' column.
    
    Args:
    messages_filepath: str. Filepath for the messages CSV file.
    categories_filepath: str. Filepath for the categories CSV file.
    
    Returns:
    df: dataframe. Dataframe obtained by merging messages and categories datasets.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean the dataframe by splitting categories into separate columns, converting to binary, and removing duplicates.
    
    Args:
    df: dataframe. Merged dataframe containing messages and categories.
    
    Returns:
    df: dataframe. Cleaned dataframe with split category columns and duplicates removed.
    """
    # Split the categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names for the categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Convert category values to just the last character (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate the original df with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataset into an SQLite database.
    
    Args:
    df: dataframe. Cleaned dataframe to be saved.
    database_filename: str. Filename for the output SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


def main():
    """
    Main function to load, clean, and save data. Runs from command line arguments.
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()