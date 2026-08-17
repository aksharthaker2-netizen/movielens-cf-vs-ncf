import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise import accuracy
from src.data_loader import load_ratings, time_based_split

def to_surprise_dataset(df, rating_scale=(1, 5)):
    """
    Convert a pandas DataFrame of (user_id, movie_id, rating) into a
    surprise Dataset object.
    """
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    return data

if __name__ == "__main__":
    ratings = load_ratings()
    train_df, test_df = time_based_split(ratings, test_frac=0.2)

    train_data = to_surprise_dataset(train_df)
    trainset = train_data.build_full_trainset()

    # Tuned hyperparameters from Day 5 grid search
    best_params = {'n_factors': 100, 'n_epochs': 30, 'lr_all': 0.01, 'reg_all': 0.1}
    model = SVD(**best_params)
    model.fit(trainset)

    testset = list(test_df[["user_id", "movie_id", "rating"]].itertuples(index=False, name=None))
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)