import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(
    BASE_DIR, "dataset", "GlucoBench_benchmark_dataset.csv"
)


def load_dataset():
    return pd.read_csv(DATASET_PATH)


def get_head(df):
    return df.head()


def get_info(df):
    return df.info()


def get_description(df):
    return df.describe()


def get_shape(df):
    return df.shape


def get_missing_values(df):
    return df.isnull().sum()


def get_tail(df):
    return df.tail()


def get_duplicates(df):
    return df.duplicated().sum()


def get_unique_values(df):
    return df.nunique()


def get_columns(df):
    return df.columns.tolist()
