import pandas as pd
import pyarrow.parquet as pq
import os
from torch.utils.data import Dataset

from config import DATA, DATA_DIR

def filter_frame(df: pd.DataFrame, col_prefixes: list) -> pd.DataFrame:
    """Filter DataFrame columns based on a list of prefixes, capturing any one-hot encoded extensions.

    Args:
        df (pd.DataFrame): The DataFrame from which to filter columns.
        col_prefixes (list): A list of column name prefixes to include in the filter.

    Returns:
        pd.DataFrame: A DataFrame containing only the columns that match the prefixes.
    """
    filtered_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in col_prefixes)]
    return df[filtered_cols]

class YelpDataset(Dataset):
    def __init__(self, user_data, item_data, label=None):
        self.user_data = user_data
        self.item_data = item_data
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        if self.label is not None:
            return self.user_data[idx], self.item_data[idx], self.label[idx]
        else:
            return self.user_data[idx], self.item_data[idx]

def update_ids_in_parquet(source_file, destination_file, chunk_size=20000):
    """ Update IDs in Parquet file to ensure continuous factorization. """
    user_id_offset = 0
    business_id_offset = 0

    parquet_file = pq.ParquetFile(source_file)
    for i in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_group(i, columns=None)
        df_chunk = chunk.to_pandas()
        print(f"Processing chunk {i}")
        # Update user IDs
        if 'review_user_id' in df_chunk:
            df_chunk['review_user_id'], uniques_user = pd.factorize(df_chunk['review_user_id'], sort=True)
            df_chunk['review_user_id'] += user_id_offset
            user_id_offset += len(uniques_user)

        # Update business IDs
        if 'gastro_business_id' in df_chunk:
            df_chunk['gastro_business_id'], uniques_business = pd.factorize(df_chunk['gastro_business_id'], sort=True)
            df_chunk['gastro_business_id'] += business_id_offset
            business_id_offset += len(uniques_business)

        # Write processed df_chunk to a new Parquet file
        if i == 0:
            df_chunk.to_parquet(destination_file, index=False, engine='fastparquet')
        else:
            df_chunk.to_parquet(destination_file, index=False, engine='fastparquet', append=True)

def chunk_to_loader(df, device, user_cols, item_cols, label_cols, seed, batch_size, validation_flag):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)

    # split in user, item, and label
    user_train_data, user_val_data = filter_frame(train_df, user_cols), filter_frame(val_df, user_cols)
    item_train_data, item_val_data = filter_frame(train_df, item_cols), filter_frame(val_df, item_cols)
    train_label, val_label = filter_frame(train_df, label_cols), filter_frame(val_df, label_cols)

    # Transform the data into PyTorch tensors
    user_train_data = torch.tensor(user_train_data.values, dtype=torch.float).to(device)
    user_val_data = torch.tensor(user_val_data.values, dtype=torch.float).to(device)
    item_train_data = torch.tensor(item_train_data.values, dtype=torch.float).to(device)
    item_val_data = torch.tensor(item_val_data.values, dtype=torch.float).to(device)
    train_label = torch.tensor(train_label.values, dtype=torch.float).to(device)
    val_label = torch.tensor(val_label.values, dtype=torch.float).to(device)

    # Create dataset and dataloader
    train_set = YelpDataset(user_train_data, item_train_data, train_label)
    val_set = YelpDataset(user_val_data, item_val_data, val_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

