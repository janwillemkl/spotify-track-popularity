"""Data preprocessing, model training, and evaluation pipeline for prediciting Spotify track popularity."""

import mlflow
import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from spotify_track_popularity.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET
from spotify_track_popularity.resources.mlflow_session import MlflowSession


@asset()
def popularity_prediction_model(
    context: AssetExecutionContext,
    mlflow_session: MlflowSession,
    train_data_balanced: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Trains an XGBoost classifier to predict track popularity.

    This function creates a machine learning pipeline to predict the popularity of Spotify tracks using an XGBoost
    classifier. Numerical features are standardized, categorical features are one-hot encoded before training the
    classifier.

    Parameters
    ----------
    context : AssetExecutionContext
        The Dagster asset execution context, which provides information about the asset and the current run.
    mlflow_session : MlflowSessions
        Session for managing MLflow experiment tracking.
    train_data_balanced : pd.DataFrame
        A DataFrame containing the balanced training data, including both features and target variable.
    test_data : pd.DataFrame
        A DataFrame containing the test data subset, including both features and target variable.
    """

    # Split features and target variable
    X_train = train_data_balanced[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_data_balanced[TARGET]

    # Define the preprocessing and model training pipeline
    numerical_pipeline = Pipeline([("encoder", StandardScaler())])
    categorical_pipeline = Pipeline([("encoder", OneHotEncoder())])

    preprocessing_pipeline = ColumnTransformer(
        [
            ("numerical_preprocessor", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical_preprocessor", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessing_pipeline),
            ("estimator", XGBClassifier()),
        ]
    )

    # Train and evaluate the model
    with mlflow_session.get_run(context):
        mlflow.xgboost.autolog()

        pipeline.fit(X_train, y_train)

        mlflow.evaluate(
            model=pipeline.predict,
            data=test_data,
            targets=TARGET,
            model_type="classifier",
        )
