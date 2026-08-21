import pandas as pd
import numpy as np
from src.data_loader import load_ratings, time_based_split
from src.svd_baseline import to_surprise_dataset

def get_top_k_recommendations(model, trainset, train_df, user_ids, k=10):
    """
    For each user, predict ratings for all movies they haven't rated in
    training, and return their top-k predicted movies.

    Returns: dict mapping user_id -> list of (movie_id, predicted_rating) tuples,
    sorted by predicted rating descending.
    """
    all_movie_ids = train_df["movie_id"].unique()
    recommendations = {}

    for user_id in user_ids:
        # Movies this user already rated in training - exclude from recommendations
        rated_movies = set(train_df[train_df["user_id"] == user_id]["movie_id"])
        candidate_movies = [m for m in all_movie_ids if m not in rated_movies]

        # Predict rating for every candidate movie
        predictions = [
            (movie_id, model.predict(user_id, movie_id).est)
            for movie_id in candidate_movies
        ]

        # Sort by predicted rating, descending, take top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations[user_id] = predictions[:k]

    return recommendations
def precision_recall_at_k(recommendations, test_df, k=10, rating_threshold=4.0):
    """
    Compute precision@k and recall@k, averaged across all users.

    A movie is considered "relevant" (liked) if the user's true test rating
    for it is >= rating_threshold.
    """
    precisions = []
    recalls = []

    for user_id, recs in recommendations.items():
        recommended_movie_ids = set(movie_id for movie_id, _ in recs)

        # All movies this user actually liked in the test set
        user_test = test_df[test_df["user_id"] == user_id]
        relevant_movies = set(user_test[user_test["rating"] >= rating_threshold]["movie_id"])

        if len(relevant_movies) == 0:
            continue  # can't compute recall if user liked nothing in test - skip

        hits = len(recommended_movie_ids & relevant_movies)

        precision = hits / k
        recall = hits / len(relevant_movies)

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)
def get_popularity_baseline_recommendations(train_df, user_ids, k=10):
    """
    Non-personalized baseline: recommend the k globally most-rated movies
    to every user (excluding movies they've already rated in training).
    """
    # Rank movies by how many times they were rated in training (proxy for popularity)
    movie_popularity = train_df["movie_id"].value_counts()
    most_popular = movie_popularity.index.tolist()  # already sorted, most popular first

    recommendations = {}
    for user_id in user_ids:
        rated_movies = set(train_df[train_df["user_id"] == user_id]["movie_id"])
        candidates = [m for m in most_popular if m not in rated_movies]
        # Fake a predicted-rating value (just rank position) to keep the same tuple format
        recommendations[user_id] = [(m, 0) for m in candidates[:k]]

    return recommendations
if __name__ == "__main__":
    from surprise import SVD

    ratings = load_ratings()
    train_df, test_df = time_based_split(ratings, test_frac=0.2)

    train_data = to_surprise_dataset(train_df)
    trainset = train_data.build_full_trainset()

    best_params = {'n_factors': 150, 'n_epochs': 40, 'lr_all': 0.005, 'reg_all': 0.05}
    model = SVD(**best_params)
    model.fit(trainset)

        # Quick sanity check: predicted ratings for a few (user, movie) pairs
    sample_checks = test_df.sample(5, random_state=42)
    for _, row in sample_checks.iterrows():
        pred = model.predict(row["user_id"], row["movie_id"]).est
        print(f"User {row['user_id']}, Movie {row['movie_id']}: predicted {pred:.2f}, actual {row['rating']}")

    all_users = test_df["user_id"].unique()
    print(f"Evaluating on {len(all_users)} users...")

    # SVD recommendations
    svd_recs = get_top_k_recommendations(model, trainset, train_df, all_users, k=10)
    svd_precision, svd_recall = precision_recall_at_k(svd_recs, test_df, k=10)
    print(f"SVD       - Precision@10: {svd_precision:.4f}, Recall@10: {svd_recall:.4f}")

    # Popularity baseline
    pop_recs = get_popularity_baseline_recommendations(train_df, all_users, k=10)
    pop_precision, pop_recall = precision_recall_at_k(pop_recs, test_df, k=10)
    print(f"Popularity - Precision@10: {pop_precision:.4f}, Recall@10: {pop_recall:.4f}")