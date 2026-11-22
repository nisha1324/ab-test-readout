"""Load the committed CSV and run the SQL in sql/ against it with DuckDB."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from . import source

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
COLUMNS = ["experiment_id", "variant_id", "metric_id", "time_since_start",
           "count_c", "count_t", "mean_c", "mean_t", "variance_c", "variance_t"]


def read_raw(path: Path | None = None) -> pd.DataFrame:
    """The source table, unchanged apart from column types.

    One row = one experiment x treatment arm x metric x snapshot time, holding
    the user count, mean and variance of the control group (`_c`) and of that
    treatment group (`_t`), cumulative since the start of the experiment.
    """
    df = pd.read_csv(path or source.RAW_PATH, dtype={"experiment_id": "string"})
    if list(df.columns) != COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    df["experiment_id"] = df["experiment_id"].astype(object)
    return df


def connect(raw: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB database with the raw table registered as `snapshots`."""
    con = duckdb.connect()
    con.register("snapshots", raw)
    return con


def run_sql(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Run sql/<name>.sql and return the result."""
    return con.execute((SQL_DIR / f"{name}.sql").read_text()).df()
