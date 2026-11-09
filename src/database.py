import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pipeline.db"


def save_table(df, name):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(name, conn, if_exists="replace", index=False)
    conn.close()


def query(sql):
    conn = sqlite3.connect(DB_PATH)
    result = pd.read_sql(sql, conn)
    conn.close()
    return result
