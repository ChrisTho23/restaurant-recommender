import pandas as pd
import pyarrow.parquet as pq
import os

from config import DATA, DATA_DIR

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

