# restaurant-recommender

Note: Work in progress...
TODO's:
- 

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
