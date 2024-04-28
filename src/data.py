import argparse
import os
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder

from config import DATA, DATA_DIR, COLS, DROP_COLS, GOOGLE_CLOUD_CREDENTIALS
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_CREDENTIALS

def load_raw_datasets(data_dir):
    """Download data from Kaggle."""
    import kaggle
    logging.info("Downloading data from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset='yelp-dataset/yelp-dataset', 
        path=data_dir,
        unzip=True,
    )
    logging.info("Data downloaded successfully!")

def preprocess_chunk(df, drop_cols):
    """Preprocess a chunk of the dataframe."""
    df = df.dropna(subset=["review_stars", "gastro_name"])
    df['user_elite'] = df['user_elite'].fillna('Never')
    df = df.drop(columns=drop_cols)

    # Label encoding review_user_id and gastro_business_id bc too many unique values
    df['review_user_id'], _ = pd.factorize(df['review_user_id'])
    df['gastro_business_id'], _ = pd.factorize(df['gastro_business_id'])

    for column in ['gastro_categories', 'user_elite']:
        dummies = df[column].str.get_dummies(sep=',').astype(int)
        dummies.columns = [f"{column}_{cat}" for cat in dummies.columns]
        df = pd.concat([df, dummies], axis=1)
        df.drop(column, axis=1, inplace=True)

    # Automatically identify and encode remaining categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols).astype(int)
    
    return df

def add_missing_categories(chunk, all_categories):
    missing_categories = [cat for cat in all_categories if cat not in chunk.columns]
    if missing_categories:
        missing_df = pd.DataFrame(0, index=chunk.index, columns=missing_categories)
        chunk = pd.concat([chunk, missing_df], axis=1)

    return chunk

def load_final_dataset(
        data_dir, destination_file, col_names, drop_cols, chunk_size, num_chunks
    ):
    """Downloads a blob from the bucket and processes data."""
    logging.info("Downloading data from the bucket...")
    storage_client = storage.Client(project="restaurant-recommender")
    bucket = storage_client.bucket("restaurant-recommender-dataset")
    blob = bucket.blob("out-s0.csv.gz")
    blob.download_to_filename(data_dir / "data.csv.gz")

    all_categories = set()
    logging.info("First processing block...")
    df = pd.read_csv(
        data_dir / "data.csv.gz", compression='gzip', escapechar='\\', delimiter="\t",
        names=col_names, chunksize=chunk_size, parse_dates=["review_date", "user_yelping_since"]
    )
    count = 0
    for chunk in df:
        logging.info(f"Processing chunk {count}")
        if num_chunks is not None and count == num_chunks:
            break
        processed_chunk = preprocess_chunk(chunk, drop_cols)
        all_categories.update(set(processed_chunk.columns))
        logging.info(f"# of columns of chunk {count}: {processed_chunk.shape[1]}...")
        count += 1
    sorted_categories = sorted(list(all_categories))

    logging.info("Second processing block...")
    df = pd.read_csv(
        data_dir / "data.csv.gz", compression='gzip', escapechar='\\', delimiter="\t",
        names=col_names, chunksize=chunk_size, parse_dates=["review_date", "user_yelping_since"]
    )
    count = 0
    for chunk in df:
        logging.info(f"Processing chunk {count}")
        if num_chunks is not None and count == num_chunks:
            break
        processed_chunk = preprocess_chunk(chunk, drop_cols)
        processed_chunk = add_missing_categories(processed_chunk, all_categories)
        processed_chunk = processed_chunk[sorted_categories]
        if count == 0:
            processed_chunk.to_parquet(destination_file, index=False, engine='fastparquet')
        else:
            processed_chunk.to_parquet(
                DATA_DIR / 'train.parquet', index=False, engine='fastparquet', append=True
            )
        count += 1

    # Clean up the downloaded file
    os.unlink(data_dir / "data.csv.gz")
    logging.info("Data processed and saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process data.')
    parser.add_argument('--raw_data', action='store_true', help='Download the raw data from Kaggle.')
    parser.add_argument('--final_data', action='store_true', help='Download the final dataset from the bucket.')
    parser.add_argument('--num_chunks', type=int, default=None, help='Number of chunks to load and process.')
    args = parser.parse_args()

    if os.path.exists(DATA["train"]):
        os.remove(DATA["train"])

    if args.final_data:
        load_final_dataset(DATA_DIR, DATA["train"], COLS, DROP_COLS, 20000, args.num_chunks)
    elif args.raw_data:
        load_raw_datasets(DATA_DIR)
