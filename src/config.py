from pathlib import Path    

DATA_DIR = Path('../data')
MODEL_DIR = Path('../model')

# dict with the names of the files and their respective paths
DATA = {
    'business': DATA_DIR / 'yelp_academic_dataset_business.json',
    'review': DATA_DIR / 'yelp_academic_dataset_review.json',
    'user': DATA_DIR / 'yelp_academic_dataset_user.json',
    'agreement': DATA_DIR / 'Dataset_User_Agreement.pdf',
    'checkin': DATA_DIR / 'yelp_academic_dataset_checkin.json',
    'tip': DATA_DIR / 'yelp_academic_dataset_tip.json',
    'train': DATA_DIR / 'train.parquet',
    'inference': DATA_DIR / 'inference.parquet',
    'predictions': DATA_DIR / 'predictions.parquet',
}

# Path to the Google Cloud credentials
GOOGLE_CLOUD_CREDENTIALS = "../application_default_credentials.json"

# Columns in dataset
COLS = [
    "gastro_business_id", "gastro_name", "gastro_address", "gastro_city", "gastro_state", 
    "gastro_postal_code", "gastro_latitude", "gastro_longitude", "gastro_stars", "gastro_review_count", 
    "gastro_categories", "review_id", "review_user_id", "review_stars", "review_useful", "review_funny", 
    "review_cool", "review_text", "review_date", "user_name", "user_review_count", "user_yelping_since", 
    "user_useful", "user_funny", "user_cool", "user_elite", "user_friends", "user_fans", 
    "user_average_stars"
]

DROP_COLS = [
    'gastro_name', 'gastro_address', 'gastro_latitude', 'gastro_longitude', 'review_id', 
    'review_text', 'review_date', 'user_name', 'user_yelping_since', 'user_friends',
    'gastro_postal_code'
]

# Columns containing user data
USER_COLS = [
    'review_user_id', 'review_useful', 'review_funny', 'review_cool', 'user_review_count', 
    'user_useful', 'user_funny', 'user_cool', 'user_elite', 'user_fans', 'user_average_stars'
]

# Columns containing item data
ITEM_COLS = [
    'gastro_business_id', 'gastro_city', 'gastro_state', 'gastro_stars', 
    'gastro_review_count', 'gastro_categories'
]

LABEL_COLS = ['review_stars']

CAT_COLS = [
    "gastro_business_id", "gastro_city", "gastro_state", "gastro_categories", 
    "review_user_id", "user_elite"
]

MODEL = {
    'NCF': MODEL_DIR / 'ncf.pth',
    'embedding_dim': 128,
    'dropout': 0.2,
}

TRAIN = {
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
}

TRAIN['chunk_size'] = 3000 * TRAIN['batch_size']

SEED = 42