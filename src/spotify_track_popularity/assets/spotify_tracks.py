"""Download and load the Spotify tracks dataset."""

import pandas as pd
from dagster import asset

from spotify_track_popularity.resources.kagge_dataset_downloader import KaggleDatasetDownloader


@asset()
def spotify_tracks(kaggle_dataset_downloader: KaggleDatasetDownloader) -> pd.DataFrame:
    """Downloads the Spotify tracks dataset.

    Parameters
    ----------
    kaggle_dataset_downloader : KaggleDatasetDownloader
        The resource used to download the dataset from Kaggle.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Spotify tracks dataset.

    """
    return kaggle_dataset_downloader.download()
