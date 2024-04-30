import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import logging
import pandas as pd
import time
import wandb
import argparse
 
from config import DATA, DATA_DIR, MODEL, TRAIN, USER_COLS, ITEM_COLS, LABEL_COLS, MODEL_DIR, SEED
from model import NCF
from utils import chunk_to_loader, YelpDataset, filter_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, train_loader, optimizer, epoch):
    running_loss = 0.
    batch_loss = []

    for batch, data in enumerate(train_loader):
        user_data, item_data, label = data

        optimizer.zero_grad()
        _, train_loss = model(user_data, item_data, label)
        train_loss.backward()
        optimizer.step()

        running_loss += train_loss.item()
        if batch % 1000 == 999:
            batch_loss.append(running_loss / 1000) # loss per batch
            logging.info(f"Epoch: {epoch}, Batch: {batch}, Avg. loss across lass 1000 batches: {batch_loss[-1]}")
            running_loss = 0.
        
    return sum(batch_loss) / len(batch_loss) if batch_loss else (running_loss / (batch + 1))

@torch.no_grad()
def evaluate(model, val_loader, epoch):
    running_loss = 0.

    for i, val_data in enumerate(val_loader):
        user_data, item_data, label = val_data

        _, loss = model(user_data, item_data, label)

        running_loss += loss.item()

    avg_vloss = running_loss / (i + 1)
    logging.info(f"Validation loss in epoch {epoch}: {avg_vloss}")

    return avg_vloss

if __name__ == '__main__':
    logging.info("Training NCF model")
    
    # Set seed
    torch.manual_seed(SEED)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train NCF model.')
    parser.add_argument(
        "--wandb_mode", help="Set the mode for wandb.", 
        choices=["online", "offline", "disabled"], default="disabled"
    )
    parser.add_argument(
        '--chunk_size', type=int, default=TRAIN["chunk_size"], help='Size of each chunk to load and process.' 
    ) 
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="restuarant-recommender", mode=args.wandb_mode)

    # Set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Get user and item shape to build model
    parquet_file = pq.ParquetFile(DATA["train"])
    schema = parquet_file.schema.to_arrow_schema()
    user_dim = len([field.name for field in schema if any(field.name.startswith(prefix) for prefix in USER_COLS)])
    item_dim = len([field.name for field in schema if any(field.name.startswith(prefix) for prefix in ITEM_COLS)])

    logging.info(f"User data has {user_dim} columns and item data has {item_dim} columns.")

    # Build the model
    model = NCF(
        user_dim=user_dim,
        item_dim=item_dim,
        embedding_dim=MODEL["embedding_dim"], 
        dropout=MODEL["dropout"]
    ).to(device)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=TRAIN["lr"])

    # Log run
    wandb.config.update({
        "batch_size": TRAIN["batch_size"],
        "learning_rate": TRAIN["lr"],
        "embedding_dim": MODEL["embedding_dim"],
        "dropout": MODEL["dropout"],
        "epochs": TRAIN["epochs"]
    })

    # Model training
    epoch_loss = {
        "train": [],
        "val": []
    }

    for epoch in range(TRAIN["epochs"]):
        logging.info(f"Epoch: {epoch}")
        chunk_loss = {
            "train": [],
            "val": []
        }

        # train and evlauate model on each chunk
        for i in range(parquet_file.num_row_groups):
            logging.info(f"Processing chunk {i}/{parquet_file.num_row_groups }")
            chunk = parquet_file.read_row_group(i, columns=None)
            df_chunk = chunk.to_pandas()
            chunk_train_loader, chunk_val_loader = chunk_to_loader(
                df_chunk, device, USER_COLS, ITEM_COLS, LABEL_COLS, SEED, TRAIN["batch_size"]
            )
            model.train()
            train_loss = train(model, chunk_train_loader, optimizer, epoch)
            chunk_loss["train"].append(train_loss)

            model.eval()
            val_loss = evaluate(model, chunk_val_loader, epoch)
            chunk_loss["val"].append(val_loss)

            wandb.log({
                "chunk": i,
                "chunk_train_loss": train_loss,
                "chunk_val_loss": val_loss
            })
        
        epoch_loss["train"].append(sum(chunk_loss["train"]) / len(chunk_loss["train"])) # avg train loss
        epoch_loss["val"].append(sum(chunk_loss["val"]) / len(chunk_loss["val"])) # avg val loss

        # Log aggregated epoch losses
        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": epoch_loss["train"][-1],
            "epoch_val_loss": epoch_loss["val"][-1]
        })

        logging.info(
            f"Epoch {epoch} finished: Avg. Training Loss: {epoch_loss['train'][-1]}, " 
            f"Avg. Validation Loss: {epoch_loss['val'][-1]}"
        )

        # Save the best model
        if epoch_loss["val"][-1] == min(epoch_loss["val"]):
            torch.save(model.state_dict(), MODEL["NCF"])
            wandb.run.summary["best_val_loss"] = epoch_loss["val"][-1]

    wandb.finish()
