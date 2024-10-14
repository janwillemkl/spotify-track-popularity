"""Spotify track popularity prediction."""

from dagster import Definitions

from spotify_track_popularity.assets.popularity_prediction_model import popularity_prediction_model
from spotify_track_popularity.assets.spotify_tracks import spotify_tracks
from spotify_track_popularity.assets.spotify_tracks_features import spotify_tracks_features
from spotify_track_popularity.assets.train_data_balanced import train_data_balanced
from spotify_track_popularity.assets.train_test import train_test
from spotify_track_popularity.constants import DATASET_REFERENCE
from spotify_track_popularity.io_managers.csv_fs_io_manager import CSVFSIOManager
from spotify_track_popularity.resources.kagge_dataset_downloader import KaggleDatasetDownloader
from spotify_track_popularity.resources.mlflow_session import MlflowSession

definitions = Definitions(
    assets=[
        spotify_tracks,
        spotify_tracks_features,
        train_test,
        train_data_balanced,
        popularity_prediction_model,
    ],
    resources={
        "kaggle_dataset_downloader": KaggleDatasetDownloader(dataset=DATASET_REFERENCE),
        "mlflow_session": MlflowSession(tracking_url="http://localhost:5000", experiment="Spotify track popularity"),
        "io_manager": CSVFSIOManager(base_dir="data"),
    },
)
