from pathlib import Path    

DATA_DIR = Path('../data')

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

GOOGLE_CLOUD_CREDENTIALS = "/Users/Christophe/.config/gcloud/application_default_credentials.json"