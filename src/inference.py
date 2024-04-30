import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pyarrow.parquet as pq
import argparse
import logging

from config import DATA, MODEL, USER_COLS, ITEM_COLS
from model import NCF
from utils import chunk_to_loader, YelpDataset, filter_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_first_chunk(file_path):
    parquet_file = pq.ParquetFile(file_path)
    chunk = parquet_file.read_row_group(0, columns=None)
    df_chunk = chunk.to_pandas()
    return df_chunk

def create_random_data(input_file, output_file, user_cols, item_cols, num_lines=10):
    logging.info(f"Creating random data from {input_file} ...")
    df = load_first_chunk(input_file)
    user_data = filter_frame(df, USER_COLS)
    item_data = filter_frame(df, ITEM_COLS)
    user_sample = user_data.sample(n=num_lines, replace=True).reset_index(drop=True)
    item_sample = item_data.sample(n=num_lines, replace=True).reset_index(drop=True)
    combined_sample = pd.concat([user_sample, item_sample], axis=1)
    combined_sample.to_parquet(output_file)
    return combined_sample

if __name__ == "__main__":
    logging.info("Starting inference...")

    # Set seed
    torch.manual_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--create_data', action='store_true', help='Flag for creating random data.'
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    if args.create_data:
        df_inference = create_random_data(DATA["train"], DATA["inference"], USER_COLS, ITEM_COLS)
    else: 
        df_inference = pd.read_parquet(DATA["inference"])

    # Load data into dataset
    user_data = filter_frame(df_inference, USER_COLS)
    item_data = filter_frame(df_inference, ITEM_COLS)

    user_data = torch.tensor(user_data.values, dtype=torch.float).to(device)
    item_data = torch.tensor(item_data.values, dtype=torch.float).to(device)

    # Load model
    model = NCF(
        user_dim=user_data.shape[1],
        item_dim=item_data.shape[1],
        embedding_dim=MODEL["embedding_dim"],
        dropout=MODEL["dropout"]
    )
    model.load_state_dict(torch.load(MODEL["NCF"], map_location=device))
    model.to(device)
    model.eval()

    # Perform inference
    all_predictions = []
    with torch.no_grad():
        predictions, _ = model(user_data, item_data)
        all_predictions.extend(predictions.cpu().numpy())
    
    # log the predictions
    logging.info(f"First five Predictions: {[pred[0] for pred in all_predictions[:5]]}")

    # Save predictions to a parquet file
    predictions_df = pd.DataFrame(all_predictions, columns=['predictions'])
    predictions_df.to_parquet(DATA["predictions"])