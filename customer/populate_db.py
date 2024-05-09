import getpass
import os
import shutil
import sqlite3

import pandas as pd
import requests
from dotenv import load_dotenv


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def populate():
    _set_env("ANTHROPIC_API_KEY")
    _set_env("TAVILY_API_KEY")

    # Recommended
    _set_env("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"

    db_url = (
        "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    )
    local_file = "travel2.sqlite"
    # The backup lets us restart for each tutorial section
    backup_file = "travel2.backup.sqlite"
    overwrite = False
    if overwrite or not os.path.exists(local_file):
        response = requests.get(db_url)
        response.raise_for_status()  # Ensure the request was successful
        with open(local_file, "wb") as f:
            f.write(response.content)
        # Backup - we will use this to "reset" our DB in each section
        shutil.copy(local_file, backup_file)
    # Convert the flights to present time for our tutorial
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()

    db = local_file  # We'll be using this local file as our DB in this tutorial
    print("Database populated successfully.")

    return db


def get_db(db_name="travel2.sqlite"):
    local_file = db_name
    # Convert the flights to present time for our tutorial
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()
    return conn, cursor


if __name__ == "__main__":
    load_dotenv()
    # Check if the required environment variables are set
    try:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        tavily_api_key = os.environ["TAVILY_API_KEY"]
        langsmith_api_key = os.environ["LANGSMITH_API_KEY"]
        populate()
    except KeyError as e:
        print(f"Error: {e.args[0]} environment variable is not set.")
        exit(1)
