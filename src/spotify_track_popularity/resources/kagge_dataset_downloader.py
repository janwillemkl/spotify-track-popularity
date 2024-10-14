"""Download datasets from Kaggle."""

import glob
import os
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd
from dagster import ConfigurableResource


class KaggleDatasetDownloader(ConfigurableResource):
    """Downloads datasets from Kaggle.

    Attributes
    ----------

    dataset : str
        The reference to the Kaggle dataset to be downloaded.
    username : Optional[str]
        The Kaggle API user.
    key : Optional[str]
        The Kaggle API key.

    API credentials may be passed directly upon instantiation, through the "KAGGLE_USERNAME" and "KAGGLE_KEY"
    environment variables, or via the "~/.kaggle/kaggle.json" file.
    """

    dataset: str

    username: Optional[str]
    key: Optional[str]

    def download(self) -> pd.DataFrame:
        """Downloads the specified dataset from Kaggle and loads it into a DataFrame.

        The dataset is downloaded, unzipped, and loaded from the first CSV file found in the dataset.

        Returns
        -------
        pd.DataFrame
            The downloaded dataset loaded into a DataFrame.
        """

        # Set API credentials when provided.
        if self.username:
            os.environ["KAGGLE_USERNAME"] = self.username
        if self.key:
            os.environ["KAGGLE_KEY"] = self.key

        # API credentials have to be known at import time.
        from kaggle import KaggleApi

        api = KaggleApi()
        api.authenticate()

        with TemporaryDirectory() as temporary_dictionary:
            api.dataset_download_files(
                dataset=self.dataset,
                path=temporary_dictionary,
                unzip=True,
            )

            # Find the CSV file in the dataset directory. Note that this only works for datasets consisting of single
            # CSV files.
            csv_files = glob.glob(f"{temporary_dictionary}/*.csv")
            if not csv_files:
                raise FileNotFoundError("No CSV file found in the downloaded dataset.")
            dataset_path = csv_files[0]

            return pd.read_csv(dataset_path)
