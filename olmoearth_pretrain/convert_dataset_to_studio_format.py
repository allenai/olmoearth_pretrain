"""Utility script for converting datasets to Studio format."""

from datetime import datetime

import pandas as pd

path = "/Users/henryh/Desktop/mangrove_100k.csv"
START_COLUMN_NAME = "start_time"
END_COLUMN_NAME = "end_time"

TASK_NAME_COLUMN_NAME = "task_name"

TASK_NAME = "mangrove_classification"

CLASS_COLUMN_NAME = "ref_cls"
MANGROVE_START_TIME = datetime(2020, 1, 1)
MANGROVE_END_TIME = datetime(2020, 12, 31)

path2 = "/Users/henryh/Desktop/mangrove_100k_studio.csv"
df = pd.read_csv(path2)
df = df.sample(100000, random_state=42)

print(df[CLASS_COLUMN_NAME].value_counts())
# # make path3 name mangrove_10_2_100k_studio.csv
# path3 = "/Users/henryh/Desktop/mangrove_10_2_100k_studio.csv"
# df.to_csv(path3, index=False)

# MANGROVE_CLASSES = {1: "Mangrove", 2: "Water", 3: "Other"}
# df = pd.read_csv(path)
# # Add a new column for the task name
# df[TASK_NAME_COLUMN_NAME] = TASK_NAME

# # Add a new column for the start time
# df[START_COLUMN_NAME] = MANGROVE_START_TIME


# # Add a new column for the end time
# df[END_COLUMN_NAME] = MANGROVE_END_TIME


# print(df.head())
# # map the class column to the names
# df[CLASS_COLUMN_NAME] = df[CLASS_COLUMN_NAME].map(MANGROVE_CLASSES)

# print(df.head())
# # write to a new csv with _studio appended to the end
# import os
# base, _ = os.path.splitext(path)
# # print the column names
# print(df.columns)
# df.to_csv(f"{base}_studio.csv", index=False)
