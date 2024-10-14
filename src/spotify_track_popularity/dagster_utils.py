"""Utility functions for handling the Dagster asset execution context."""

from dagster import AssetExecutionContext

SHORT_RUN_ID_LENGTH = 8
"""Number of characters to include in the shortened version of the run ID."""


def get_run_id(context: AssetExecutionContext, short: bool = False) -> str:
    """Retrieves the run ID from the Dagster execution context, optionally as shortened version.

    Parameters
    ----------
    context : AssetExecutionContext
        The Dagster asset execution context, which provides information about the current run.
    short : bool, optional, default False
        If True, this function returns only the first `SHORT_RUN_ID_LENGTH` characters of the run ID.

    Returns
    -------
    str
        The run ID for the current execution. Returns a shortened version of `short` is True.
    """
    run_id = context.run.run_id
    return run_id[:SHORT_RUN_ID_LENGTH] if short else run_id


def get_asset_key(context: AssetExecutionContext) -> str:
    """Converts the asset keys to a user-readable string format.

    Parameters
    ----------
    context : AssetExecutionContext
        The Dagster asset execution context, which provides information about the asset key.

    Returns
    -------
    str
        A string representation of the asset key.
    """
    return context.asset_key.to_user_string()
