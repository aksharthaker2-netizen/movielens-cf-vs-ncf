import pandas as pd

def load_ratings(path="data/raw/ratings.dat"):
    """
    Load ratings.dat: UserID::MovieID::Rating::Timestamp
    """
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",          # needed because '::' is multi-character; the fast C engine only supports single-character separators
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    return ratings

def load_movies(path="data/raw/movies.dat"):
    """
    Load movies.dat: MovieID::Title::Genres
    """
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )
    return movies

def load_users(path="data/raw/users.dat"):
    """
    Load users.dat: UserID::Gender::Age::Occupation::Zip-code
    """
    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1"
    )
    return users
def time_based_split(ratings, test_frac=0.2):
    """
    Per-user time-based split: for each user, their earliest (1 - test_frac)
    ratings (by timestamp) go to train, their most recent test_frac go to test.
    This guarantees every test user has training history, avoiding the
    "user only in test" problem a global timestamp cutoff can cause.
    """
    def split_user(group):
        group_sorted = group.sort_values("timestamp")
        cutoff = int(len(group_sorted) * (1 - test_frac))
        return group_sorted.iloc[:cutoff], group_sorted.iloc[cutoff:]

    train_parts = []
    test_parts = []

    for _, group in ratings.groupby("user_id"):
        train_part, test_part = split_user(group)
        train_parts.append(train_part)
        test_parts.append(test_part)

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)

    return train, test

if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()

    print("Ratings shape:", ratings.shape)
    print(ratings.head())
    print()
    print("Movies shape:", movies.shape)
    print(movies.head())
    print()
    print("Users shape:", users.shape)
    print(users.head())
    print()
    print("--- Ratings sanity checks ---")
    print("Rating value range:", ratings["rating"].min(), "-", ratings["rating"].max())
    print("Unique users:", ratings["user_id"].nunique())
    print("Unique movies rated:", ratings["movie_id"].nunique())
    print("Any missing values?")
    print(ratings.isnull().sum())
    print()
    print("--- Timestamp range ---")
    print("Earliest rating:", pd.to_datetime(ratings["timestamp"].min(), unit="s"))
    print("Latest rating:", pd.to_datetime(ratings["timestamp"].max(), unit="s"))
    print()
    print("--- Time-based split ---")
    train, test = time_based_split(ratings, test_frac=0.2)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Train date range:", pd.to_datetime(train["timestamp"].min(), unit="s"), "to", pd.to_datetime(train["timestamp"].max(), unit="s"))
    print("Test date range:", pd.to_datetime(test["timestamp"].min(), unit="s"), "to", pd.to_datetime(test["timestamp"].max(), unit="s"))
    print()
    print("--- Cold-start check: test users/movies not seen in train ---")
    train_users = set(train["user_id"].unique())
    train_movies = set(train["movie_id"].unique())
    test_users = set(test["user_id"].unique())
    test_movies = set(test["movie_id"].unique())
    unseen_users = test_users - train_users
    unseen_movies = test_movies - train_movies
    print("Test users not in train:", len(unseen_users), "/", len(test_users))
    print("Test movies not in train:", len(unseen_movies), "/", len(test_movies))