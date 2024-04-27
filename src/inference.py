import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import argparse

from config import DATA, MODEL, USER_COLS, ITEM_COLS
from train import filter_frame, YelpDataset
from model import NCF

def load_data(file_path):
    df = pd.read_parquet(file_path, engine='pyarrow')
    return df

def create_random_data(input_file, output_file, num_lines=5):
    print(f"Creating random data from {input_file} ...")
    df = load_data(input_file)
    random_rows = df.sample(n=num_lines)
    random_rows.to_parquet(output_file)

if __name__ == "__main__":
    print("Starting inference...")

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
    print(f"Using device: {device}")

    # Load data
    if args.create_data:
        create_random_data(DATA["train"], DATA["inference"], 5)
    df = load_data(DATA["inference"])
    user_data = filter_frame(df, USER_COLS)
    item_data = filter_frame(df, ITEM_COLS)

    user_data = torch.tensor(user_data.values).to(device)
    item_data = torch.tensor(item_data.values).to(device)

    dataset = YelpDataset(user_data, item_data)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

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
        for user_data, item_data in data_loader:
            predictions = model(user_data, item_data)
            all_predictions.extend(predictions.cpu().numpy())
    
    # Optionally, handle or save the predictions
    print("Predictions:", all_predictions)

    # Save predictions to a parquet file
    predictions_df = pd.DataFrame(all_predictions, columns=['predictions'])
    predictions_df.to_parquet(DATA["predictions"])
