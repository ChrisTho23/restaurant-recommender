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
    'data': DATA_DIR / 'data.parquet',
}

# Path to the Google Cloud credentials
GOOGLE_CLOUD_CREDENTIALS = "/Users/Christophe/.config/gcloud/application_default_credentials.json"

# Columns containing user data
USER_COLS = [
    'review_useful', 'review_funny', 'review_cool', 'review_user_id' 'user_review_count', 'user_useful', 
    'user_funny', 'user_cool', 'user_elite', 'user_fans', 'user_average_stars'
]

# Columns containing item data
ITEM_COLS = [
    'gastro_business_id', 'gastro_city', 'gastro_state', 'gastro_postal_code', 'gastro_stars', 
    'gastro_review_count', 'gastro_categories'
]

LABEL_COLS = ['review_stars']

MODEL = {
    'embedding_dim': 128,
    'dropout': 0.2,
}

TRAIN = {
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
}