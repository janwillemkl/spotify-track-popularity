"""Project configuration and constants."""

DATASET_REFERENCE = "amitanshjoshi/spotify-1million-tracks"

POPULARITY_THRESHOLD = 50

NUMERICAL_FEATURES = [
    "danceability",
    "loudness",
    "energy",
    "tempo",
    "valence",
    "speechiness",
    "liveness",
    "acousticness",
    "instrumentalness",
    "duration_ms",
    "year",
]

CATEGORICAL_FEATURES = [
    "genre",
]

TARGET = "verdict"

RANDOM_STATE = 2504

TEST_SIZE = 0.25
