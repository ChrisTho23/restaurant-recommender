import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import wandb
import argparse
 
from config import DATA, MODEL, TRAIN, USER_COLS, ITEM_COLS, LABEL_COLS, MODEL_DIR
from model import NCF

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
            print(f"Epoch: {epoch}, Batch: {batch}, Avg. loss across lass 1000 batches: {batch_loss[-1]}")
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
    print(f"Validation loss in epoch {epoch}: {avg_vloss}")

    return avg_vloss

if __name__ == '__main__':
    # Set seed
    torch.manual_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(
        "--wandb_mode", help="Set the mode for wandb.", 
        choices=["online", "offline", "disables"], default="disabled"
    )
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="restuarant-recommender", mode=args.wandb_mode)

    # Set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    df = pd.read_parquet(DATA["train"], engine='pyarrow')
    n_samples = df.shape[0]
    print(f"The dataset encompasses {n_samples} samples.")

    # Split into train and validation data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Split into user and item data
    user_train_data, user_val_data = filter_frame(train_df, USER_COLS), filter_frame(val_df, USER_COLS)
    item_train_data, item_val_data = filter_frame(train_df, ITEM_COLS), filter_frame(val_df, ITEM_COLS)
    train_label, val_label = filter_frame(train_df, LABEL_COLS), filter_frame(val_df, LABEL_COLS)
    print(
        f"The user data encompasses {user_train_data.shape[1]} features (encdoed).",
        f"\nThe item data encompasses {item_train_data.shape[1]} features (encdoed).",
        f"\nThe label data encompasses {train_label.shape[1]} features."
    )

    # Transform the data into PyTorch tensors
    user_train_data = torch.tensor(user_train_data.values).to(device)
    user_val_data = torch.tensor(user_val_data.values).to(device)
    item_train_data = torch.tensor(item_train_data.values).to(device)
    item_val_data = torch.tensor(item_val_data.values).to(device)
    train_label = torch.tensor(train_label.values).to(device)
    val_label = torch.tensor(val_label.values).to(device)

    # Create dataset and dataloader
    train_set = YelpDataset(user_train_data, item_train_data, train_label)
    val_set = YelpDataset(user_val_data, item_val_data, val_label)
    train_loader = DataLoader(train_set, batch_size=TRAIN["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=TRAIN["batch_size"], shuffle=False)
    print('Training set has {} instances'.format(len(train_set)))
    print('Validation set has {} instances'.format(len(val_set)))

    # Build the model
    model = NCF(
        user_dim=user_train_data.shape[1],
        item_dim=item_train_data.shape[1],
        embedding_dim=MODEL["embedding_dim"], 
        dropout=MODEL["dropout"]
    ).to(device)

    # Define optimizer
    optimizer = Adam(model.parameters(), lr=TRAIN["lr"])

    wandb.config.update({
        "n_samples": n_samples,
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
        print(f"Epoch: {epoch}")

        # Train the model
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch)
        epoch_loss["train"].append(train_loss)

        # Evaluate the model
        model.eval()
        val_loss = evaluate(model, val_loader, epoch)
        epoch_loss["val"].append(val_loss)

        wandb.log({"epoch_train_loss": train_loss, "epoch_val_loss": val_loss, "epoch": epoch})

        print(
            f"Training loss epoch {epoch}: {epoch_loss['train'][-1]}"
            f"\nValidation loss epoch {epoch}: {epoch_loss['val'][-1]}"
        )  

        # Save the best model
        if epoch_loss["val"][-1] == min(epoch_loss["val"]):
            torch.save(model.state_dict(), MODEL["NCF"])
            wandb.run.summary["best_val_loss"] = val_loss

    wandb.finish()
