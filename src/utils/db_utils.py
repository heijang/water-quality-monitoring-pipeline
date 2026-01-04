import os
import logging
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def get_db_url(db_name: str = "water_quality_db") -> str:
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "airflow")
    password = os.getenv("DB_PASSWORD", "airflow")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"


def get_engine(db_name: str = "water_quality_db") -> Engine:
    url = get_db_url(db_name)
    engine = create_engine(url, pool_pre_ping=True)
    logger.info(f"Created engine: {db_name}")
    return engine


def create_database_if_not_exists(db_name: str = "water_quality_db"):
    default_url = get_db_url("postgres")
    engine = create_engine(default_url, isolation_level="AUTOCOMMIT")

    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        )
        exists = result.fetchone() is not None

        if not exists:
            conn.execute(text(f"CREATE DATABASE {db_name}"))
            logger.info(f"Created database: {db_name}")
        else:
            logger.info(f"Database exists: {db_name}")

    engine.dispose()


def initialize_schema(sql_file_path: str, db_name: str = "water_quality_db"):
    engine = get_engine(db_name)

    with open(sql_file_path, 'r') as f:
        sql_script = f.read()

    with engine.connect() as conn:
        for statement in sql_script.split(';'):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
                conn.commit()

    logger.info(f"Schema initialized from {sql_file_path}")
    engine.dispose()


def load_csv_to_table(csv_path: str, table_name: str,
                      db_name: str = "water_quality_db",
                      if_exists: str = "replace") -> int:
    df = pd.read_csv(csv_path)
    engine = get_engine(db_name)

    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method='multi',
        chunksize=1000
    )

    logger.info(f"Loaded {len(df)} rows to {table_name}")
    engine.dispose()
    return len(df)


def read_table(table_name: str, db_name: str = "water_quality_db",
               query: Optional[str] = None) -> pd.DataFrame:
    engine = get_engine(db_name)

    if query is None:
        query = f"SELECT * FROM {table_name}"

    df = pd.read_sql(query, engine)
    logger.info(f"Read {len(df)} rows from {table_name}")
    engine.dispose()
    return df


def save_dataframe_to_table(df: pd.DataFrame, table_name: str,
                             db_name: str = "water_quality_db",
                             if_exists: str = "append") -> int:
    engine = get_engine(db_name)

    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method='multi',
        chunksize=1000
    )

    logger.info(f"Saved {len(df)} rows to {table_name}")
    engine.dispose()
    return len(df)


def execute_query(query: str, db_name: str = "water_quality_db") -> pd.DataFrame:
    engine = get_engine(db_name)
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


def save_data_with_mode(df: pd.DataFrame,
                        output_path: str,
                        table_name: str,
                        storage_mode: str = "LOCAL",
                        db_name: str = "water_quality_db",
                        if_exists: str = "replace") -> None:
    """
    Save DataFrame based on storage mode
    - LOCAL: Save to CSV only
    - DB: Save to database only
    - BOTH: Save to both CSV and database (read from LOCAL)
    """
    import os

    if storage_mode in ("LOCAL", "BOTH"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to CSV: {output_path}")

    if storage_mode in ("DB", "BOTH"):
        try:
            save_dataframe_to_table(df, table_name, db_name, if_exists)
            logger.info(f"Saved to DB table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to save to DB table {table_name}: {e}")
            if storage_mode == "DB":
                # If DB-only mode and DB save fails, raise error
                raise
            # For BOTH mode, CSV is already saved, so just log warning
            logger.warning("Continuing with CSV-only storage")


def load_data_with_mode(input_path: str,
                        table_name: str,
                        storage_mode: str = "LOCAL",
                        db_name: str = "water_quality_db") -> pd.DataFrame:
    """
    Load DataFrame based on storage mode
    - LOCAL: Load from CSV
    - DB: Load from database
    - BOTH: Load from CSV (DB is also updated but read from LOCAL)
    """
    if storage_mode == "DB":
        try:
            df = read_table(table_name, db_name)
            logger.info(f"Loaded from DB table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to load from DB table {table_name}: {e}")
            logger.warning(f"Falling back to CSV: {input_path}")
            df = pd.read_csv(input_path)
    else:  # LOCAL or BOTH
        df = pd.read_csv(input_path)
        logger.info(f"Loaded from CSV: {input_path}")

    return df
