"""Oversample the training data subset."""

import pandas as pd
from dagster import asset
from imblearn.over_sampling import RandomOverSampler

from spotify_track_popularity.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET


@asset()
def train_data_balanced(train_data: pd.DataFrame) -> pd.DataFrame:
    """Applies random oversampling to the training data to address class imbalance.

    In the training data, popular tracks are underrepresented, resulting in a class imbalance. Many machine learning
    methods are sensitive to imbalanced data, which can lead to biased predictions towards the majority class (i.e.,
    tracks that are not popular). This function uses `RandomOverSampler` to balance the classes by duplicating
    instances of the minority class.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training data subset, containing both the features and the target variable.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the balanced training data.
    """

    X_train = train_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_data[TARGET]

    X_train_balanced, y_train_balanced = RandomOverSampler().fit_resample(X_train, y_train)

    return X_train_balanced.join(y_train_balanced)
