import pandas as pd


def mapping_id_to_unique(df: pd.DataFrame, target_df: pd.DataFrame):
    """
    Id(UserId, MovieId)를 임베딩의 입력으로 준비.

    Args:
        df (DataFrame): 전체 데이터프레임
        target_df: 매핑 로직을 적용할 데이터프레임

    Return:
        num_users: unique users in df
        num_movies: unique movies in df
        target_df: Mapped DataFrame
    """
    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    # Mapper
    user_to_idx = {original: new for new, original in enumerate(unique_users)}
    movie_to_idx = {original: new for new, original in enumerate(unique_movies)}

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    target_df["userId"] = target_df["userId"].map(user_to_idx)
    target_df["movieId"] = target_df["movieId"].map(movie_to_idx)

    target_df.dropna(inplace=True)
    target_df["userId"] = target_df["userId"].astype(int)
    target_df["movieId"] = target_df["movieId"].astype(int)

    return num_users, num_movies, target_df
