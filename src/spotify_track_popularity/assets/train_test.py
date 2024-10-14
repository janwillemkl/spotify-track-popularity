"""Split the Spotify tracks features and popularity labels into training and testing sets."""

import pandas as pd
from dagster import AssetOut, multi_asset
from sklearn.model_selection import train_test_split

from spotify_track_popularity.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


@multi_asset(
    outs={
        "train_data": AssetOut(description="A DataFrame containing the training data subset."),
        "test_data": AssetOut(description="A DataFrame containing the test data subset."),
    }
)
def train_test(spotify_tracks_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into a training and testing sets.

    Parameters
    ----------
    spotify_tracks_features : pd.DataFrame
        A DataFrame containing the selected features and popularity label for Spotify tracks.

    Returns
    -------
    train_data : pd.DataFrame
        A DataFrame containing the training data subset.
    test_data : pd.DataFrame
        A DataFrame containing the test data subset.
    """

    train_data, test_data = train_test_split(
        spotify_tracks_features,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_data = train_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]]
    test_data = test_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]]

    return train_data, test_data
