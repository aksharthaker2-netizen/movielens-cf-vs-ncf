import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV
from src.data_loader import load_ratings, time_based_split
from src.svd_baseline import to_surprise_dataset

if __name__ == "__main__":
    ratings = load_ratings()
    train_df, test_df = time_based_split(ratings, test_frac=0.2)

    train_data = to_surprise_dataset(train_df)

    param_grid = {
        "n_factors": [50, 100, 150],
        "n_epochs": [20, 30, 40],
        "lr_all": [0.005, 0.01, 0.02],
        "reg_all": [0.02, 0.05, 0.1]
    }

    gs = GridSearchCV(
        SVD,
        param_grid,
        measures=["rmse"],
        cv=3,        # 3-fold cross-validation within the training data
        n_jobs=-1    # use all available CPU cores in parallel
    )

    gs.fit(train_data)

    print("Best RMSE:", gs.best_score["rmse"])
    print("Best params:", gs.best_params["rmse"])