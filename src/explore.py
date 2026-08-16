import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_ratings, load_movies, load_users

ratings = load_ratings()
movies = load_movies()
users = load_users()

# 1. Ratings distribution
print("--- Ratings distribution ---")
print(ratings["rating"].value_counts().sort_index())
print()
print("Mean rating:", ratings["rating"].mean().round(3))

ratings["rating"].value_counts().sort_index().plot(kind="bar", title="Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig("results/ratings_distribution.png")
plt.close()

# 2. Ratings per user
ratings_per_user = ratings.groupby("user_id").size()
print()
print("--- Ratings per user ---")
print(ratings_per_user.describe())
print("Users with fewer than 20 ratings:", (ratings_per_user < 20).sum())

# 3. Ratings per movie
ratings_per_movie = ratings.groupby("movie_id").size()
print()
print("--- Ratings per movie ---")
print(ratings_per_movie.describe())
print("Movies with fewer than 5 ratings:", (ratings_per_movie < 5).sum())