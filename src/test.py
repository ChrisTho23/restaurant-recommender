import pandas as pd

from config import DATA

if __name__ == '__main__':
    df = pd.read_parquet(DATA["data"], engine='pyarrow')
    print(
        f"These are the first 5 rows of the dataset:\n{df.head()}"
        f"\n\nThe dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    )

    # Drop samples in which review_stars is NaN
    df = df.dropna(subset=["review_stars", "gastro_name"])

    # Drop not needed columns
    df = df.drop(columns=[
        'gastro_business_id', 'gastro_latitude', 'gastro_longitude', 'review_id', 'review_user_id',
        'user_name', 'review_text', 'review_date', 'user_yelping_since', 'user_friends'
    ])

    if 'gastro_postal_code' in df.columns:
        df['gastro_postal_code'] = df['gastro_postal_code'].astype(str)

    # Print the number of missing values in each column
    print(f"\nNumber of missing values in each column befroe imputation:\n{df.isnull().sum()}")

    columns_to_split = ['gastro_categories', 'user_elite']
    encoded_cols = {}
    for column in columns_to_split:
        exploded = df[column].str.split(',').explode()
        dummies = pd.get_dummies(exploded, prefix=column)
        dummies = dummies.groupby(dummies.index).sum()
        df = df.join(dummies) 

    # Drop the original columns
    df = df.drop(columns=columns_to_split)

    # Impute NaN values
    for column in df.select_dtypes(include=['object', 'string']).columns:
        df[column] = df[column].fillna('Unknown')
    for column in df.select_dtypes(include=['int', 'float']).columns:
        df[column] = df[column].fillna(0)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols)

    # Print the number of missing values in each column
    print(f"\nNumber of missing values: {df.isnull().sum().sum()}")
    print(f"\nThese are the first 5 rows of the dataset after cleaning:\n{df.head()}")

    non_numeric_columns = df.select_dtypes(exclude=['int', 'float', 'bool']).shape[1]
    print(f"The number of columns that are not of type numeric or float is: {non_numeric_columns}")