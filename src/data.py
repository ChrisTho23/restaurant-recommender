import kaggle
import argparse
import os
import pandas as pd

from config import DATA, DATA_DIR, GOOGLE_CLOUD_CREDENTIALS
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_CREDENTIALS

def load_raw_datasets():
    print("Downloading data from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset='yelp-dataset/yelp-dataset', 
        path=DATA_DIR,
        unzip=True,
    )
    print("Data downloaded successfully!")

def load_final_dataset(destination_dir, n_rows):
    """Downloads a blob from the bucket."""
    print("Downloading data from the bucket...")

    # Download the file from the bucket
    storage_client = storage.Client(project='e-charger-418218')
    bucket = storage_client.bucket("restaurant-recommender-dataset")
    blob = bucket.blob("out-s0.csv.gz")
    blob.download_to_filename(DATA_DIR / "data.csv.gz")

    # Read the downloaded file into a pandas DataFrame
    col_names = [
        "gastro_business_id", "gastro_name", "gastro_address", "gastro_city", "gastro_state", 
        "gastro_postal_code", "gastro_latitude", "gastro_longitude", "gastro_stars", "gastro_review_count", 
        "gastro_categories", "review_id", "review_user_id", "review_stars", "review_useful", "review_funny", 
        "review_cool", "review_text", "review_date", "user_name", "user_review_count", "user_yelping_since", 
        "user_useful", "user_funny", "user_cool", "user_elite", "user_friends", "user_fans", 
        "user_average_stars"
    ]
    df = pd.read_csv(
        DATA_DIR / "data.csv.gz", names=col_names, compression='gzip', escapechar='\\', delimiter="\t", 
        parse_dates= ["review_date", "user_yelping_since"], index_col=False, nrows=n_rows
    )
    print(
        f"These are the first 5 rows of the dataset:\n{df.head()}"
        f"\n\nThe dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    )

    # Drop samples in which review_stars is NaN
    df = df.dropna(subset=["review_stars", "gastro_name"])

    # Drop not needed columns
    df = df.drop(columns=[
        'gastro_name', 'gastro_address', 'gastro_latitude', 'gastro_longitude', 'review_id', 
        'user_name', 'review_text', 'review_date', 'user_yelping_since', 'user_friends'
    ])

    if 'gastro_postal_code' in df.columns:
        df['gastro_postal_code'] = df['gastro_postal_code'].astype(str)

    # Print the number of missing values in each column
    print(f"\nNumber of missing values in each column befroe imputation:\n{df.isnull().sum()}")

    columns_to_split = ['gastro_categories', 'user_elite']
    for column in columns_to_split:
        exploded = df[column].str.split(',').explode()
        dummies = pd.get_dummies(exploded, prefix=column).astype('float32')
        dummies = dummies.groupby(dummies.index).sum()
        df = df.join(dummies) 

    # Drop the original columns
    df = df.drop(columns=columns_to_split)

    # Impute NaN values
    for column in df.select_dtypes(include=['object', 'string']).columns:
        df[column] = df[column].fillna('Unknown')
    for column in df.select_dtypes(include=['int', 'float']).columns:
        df[column] = df[column].fillna(0)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols).astype('float32')

    # Print the number of missing values in each column
    print(f"\nNumber of missing values: {df.isnull().sum().sum()}")
    print(f"\nThese are the first 5 rows of the dataset after cleaning:\n{df.head()}")

    non_numeric_columns = df.select_dtypes(exclude=['int', 'float']).shape[1]
    print(f"The number of columns that are not of type numeric or float is: {non_numeric_columns}")

    # Save the DataFrame as a parquet file
    df.to_parquet(DATA["data"], index=False, engine="pyarrow")

    (DATA_DIR / "data.csv.gz").unlink()

    print("Data downloaded successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download data.')
    parser.add_argument('--raw_data', action='store_true', help='Download the raw data from Kaggle.')
    parser.add_argument('--final_data', action='store_true', help='Download the final dataset from the bucket.')
    parser.add_argument(
        '--nrows', type=int, default=None, 
        help='Number of rows to load from the final dataset. If not set, the entire dataset will be loaded.'
    )
    args = parser.parse_args()

    if args.final_data:
        load_final_dataset(DATA_DIR, args.nrows)
    elif args.raw_data:
        load_raw_datasets()
