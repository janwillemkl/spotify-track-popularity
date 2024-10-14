"""Extract features and label popularity of Spotify tracks."""

import pandas as pd
from dagster import asset

from spotify_track_popularity.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, POPULARITY_THRESHOLD, TARGET


@asset()
def spotify_tracks_features(spotify_tracks: pd.DataFrame) -> pd.DataFrame:
    """Extracts selected features and identifies popular tracks.

    Tracks with a popularity score exceeding the `POPULARITY_THRESHOLD` are labeled as popular, and this label is added
    to the dataset as a new column specified by `TARGET`.

    The dataset is filtered to include only the specified numerical and categorical features.

    Parameters
    ----------
    spotify_tracks : pd.DataFrame
        The raw Spotify tracks dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected features along with the popularity label.
    """

    # Add track popularity verdict
    spotify_tracks[TARGET] = (spotify_tracks["popularity"] >= POPULARITY_THRESHOLD).astype(int)

    # Filter selected features and target
    return spotify_tracks[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]]
