# create_db.py
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime,
    ForeignKey, MetaData, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------- Paths ----------
DATA_DIR = Path(r"D:\flatz__backend\flatZ_data")
DB_PATH  = Path(r"D:\flatz__backend\flatz.db")

USERS_CSV = DATA_DIR / "users_realistic.csv"
ITEMS_CSV = DATA_DIR / "items_realistic.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions_realistic.csv"
TOP5_CSV = DATA_DIR / "top5_recommendations.csv"

# ---------- SQLAlchemy Setup ----------
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)
Base = declarative_base(metadata=MetaData())

# ---------- Models ----------
class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=False)

class Item(Base):
    __tablename__ = "items"
    item_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    city = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    amenities = Column(Text)

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    item_id = Column(Integer, ForeignKey("items.item_id"), nullable=False)
    interaction_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=True)
    interaction = Column(String, nullable=True)

class Top5Recommendation(Base):
    __tablename__ = "top5_recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    item_id = Column(Integer, ForeignKey("items.item_id"), nullable=False)
    rank = Column(Integer, nullable=False)  # 1-5

# ---------- Helpers ----------
def assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")

def read_csv_safe(path: Path, expected_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path.name} missing columns: {missing}")
    return df[expected_cols]

def drop_tables_if_exist():
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS top5_recommendations;"))
        conn.execute(text("DROP TABLE IF EXISTS interactions;"))
        conn.execute(text("DROP TABLE IF EXISTS items;"))
        conn.execute(text("DROP TABLE IF EXISTS users;"))

def create_tables():
    Base.metadata.create_all(bind=engine)

def load_users(df: pd.DataFrame):
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["user_id"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df.to_sql("users", engine, if_exists="append", index=False, chunksize=5000, method="multi")
    return len(df)

def load_items(df: pd.DataFrame):
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["item_id", "price"]).copy()
    df["item_id"] = df["item_id"].astype(int)
    df.to_sql("items", engine, if_exists="append", index=False, chunksize=5000, method="multi")
    return len(df)

def load_interactions(df: pd.DataFrame):
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "interaction_type"]).copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df.to_sql("interactions", engine, if_exists="append", index=False, chunksize=5000, method="multi")
    return len(df)

def load_top5(df: pd.DataFrame):
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["user_id", "item_id", "rank"]).copy()
    df.to_sql("top5_recommendations", engine, if_exists="append", index=False, chunksize=5000, method="multi")
    return len(df)

def enable_sqlite_foreign_keys():
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON;"))

# ---------- Main ----------
def main():
    assert_exists(DATA_DIR, "DATA_DIR")
    assert_exists(USERS_CSV, "users_realistic.csv")
    assert_exists(ITEMS_CSV, "items_realistic.csv")
    assert_exists(INTERACTIONS_CSV, "interactions_realistic.csv")
    assert_exists(TOP5_CSV, "top5_recommendations.csv")

    drop_tables_if_exist()
    create_tables()
    enable_sqlite_foreign_keys()

    users_n = load_users(read_csv_safe(USERS_CSV, ["user_id", "name", "email"]))
    items_n = load_items(read_csv_safe(ITEMS_CSV, ["item_id", "title", "city", "price", "amenities"]))
    interactions_n = load_interactions(read_csv_safe(
        INTERACTIONS_CSV, ["user_id", "item_id", "interaction_type", "timestamp", "interaction"]
    ))
    top5_n = load_top5(read_csv_safe(TOP5_CSV, ["user_id", "item_id", "rank"]))

    print("✅ Database created and loaded!")
    print(f"Users: {users_n}, Items: {items_n}, Interactions: {interactions_n}, Top5: {top5_n}")
    print(f"📦 SQLite DB at: {DB_PATH}")

if __name__ == "__main__":
    main()
