# restaurant-recommender

Note: Work in progress...
TODO's:
- 

Data Preprocessing:
1. Load user, business, and review dataset into kaggle (manually select JSON dataformat for user dataset)
2. Filter 'categoty' column in business dataset for gastronomy businesses via keywords: "Bakeries", "Bar", "Bars", "Bistros", "Cafes", "Patisserie", "Restaurants", "Tea" (150,346 lines to 61,562 lines)
3. Merge the the business dataset containing the gastronomy business with the review dataset with an inner join on the key "business_id". We retain the columns business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, and categories from the business dataset and the columns review_id, review_user_id, review_stars, review_useful, review_funny, review_cool, review_text, and review_date from the review dataset. (out of 6,990,280 reviews, 5,062,772 lines of data remain)
4. Last but not least, enrich the data for each review of each business by the user data using another inner join, this time on the key "user_id". We retain all the columns out of the business_review dataset and add the user information: user_name, user_review_count, user_yelping_since, user_useful, user_funny, user_cool, user_elite, user_friends, user_fans, user_average_stars to each row. Out of the 5,062,772 lines of reviews, we can find user data for X lines. The final dataset contain the columns:
1  (string) gastro_business_id
2  (string) gastro_name
3  (string) gastro_address
4  (string) gastro_city
5  (string) gastro_state
6  (string) gastro_postal_code
7  (double) gastro_latitude
8  (double) gastro_longitude
9  (double) gastro_stars
10  (bigint) gastro_review_count
11  (string) gastro_categories
12  (string) review_id
13  (string) review_user_id
14  (string) review_stars
15  (string) review_useful
16  (string) review_funny
17  (string) review_cool
18  (string) review_text
19  (string) review_date
20  (string) user_name
21  (string) user_review_count
22  (string) user_yelping_since
23  (string) user_useful
24  (string) user_funny
25  (string) user_cool
26  (string) user_elite
27  (string) user_friends
28  (string) user_fans
29  (string) user_average_stars 

## Overview & Motivation

This repository is aiming at building a recommendation system able to recommend restaurants to a given user. The model used is (DLRM or Contextual Sequence Learning with Transformer)


## Acknowledgments

This repository is inspired by 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
The project is built using Python and PyTorch. We use Poetry for dependency management. 

First, you will have to clone the repository locally.
```bash
git clone https://github.com/ChrisTho23/restaurant-recommender
cd DrakeGPT
```

Then, install dependencies using Poetry:
```bash
poetry install
```

All following scripts will have to be run from the [./src](https://github.com/ChrisTho23/restaurant-recommender/tree/main/src/) folder to make sure the relative paths defined in [./src/config.py](https://github.com/ChrisTho23/restaurant-recommender/tree/main/src/config.py) work correctly. Access the [./src](https://github.com/ChrisTho23/restaurant-recommender/tree/main/src/) file like so:
```bash
cd src/
```

In this repository, we use a subset of Yelp's businesses, reviews, and user data, included in this [Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_business.json) that has been uploaded to Kaggle, are used. Thus, we need to access Kaggle’s public API to download the dataset. For this, one needs to authenticate in Kaggle using an API token. If you have not done so, follow these steps to authenticate: 

1. If not done already, create an account on [kaggle.com](https://www.kaggle.com)
2. Go to the 'Account' tab of your user profile on the Kaggle website. Click on 'Create New API Token'. This triggers the download of `kaggle.json`, a file containing your API credentials.
3. Make the credentials in the `kaggle.json`file accessible to your application. This can look like this:

```bash
mkdir ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

4. For more details and troubleshooting, visit the [official Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials).

Finally, you will have to run the [./src/setup.py](https://github.com/ChrisTho23/restaurant-recommender/tree/main/src/data.py) script to load the data in the [./data](https://github.com/ChrisTho23/myfirstGPT/tree/main/data) folder and create a train and a test data set. We use a tiny dataset from Kaggle containing lyrics of Drake song text for model training. Find the data [here](https://github.com/ChrisTho23/restaurant-recommender/tree/main/data/) after.
```bash
poetry run python data.py
```

## Results

### Training

## Usage

### Training

To train a model, run the [src/train.py](https://github.com/ChrisTho23/restaurant-recommender/tree/main/src/train.py) script.
```bash
poetry run python train.py
```

Note: After every run of train.py the model will be saved in the [./model](https://github.com/ChrisTho23/restaurant-recommender/tree/main/model.py) folder. By default, all models were trained and can be found in this folder. Running a pre-defined model will overwrite this file.

### Inference

## Dependencies
Dependencies are managed with Poetry. To add or update dependencies, use Poetry's dependency management commands.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
