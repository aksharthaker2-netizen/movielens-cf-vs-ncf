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