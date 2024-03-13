import kaggle

from config import DATA_DIR

def load_dataset():
    print("Downloading data from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset='yelp-dataset/yelp-dataset', 
        path=DATA_DIR,
        unzip=True,
    )
    print("Data downloaded successfully!")

if __name__ == "__main__":
    load_dataset()