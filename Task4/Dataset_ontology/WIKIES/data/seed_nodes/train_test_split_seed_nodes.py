import os
import pandas as pd
from sklearn.model_selection import train_test_split


def extract_categories_and_distribution(paths):
    categories_distribution = {}

    for file_path in paths:
        df = pd.read_csv(file_path)
        if 'level3_main_occ' in df.columns:
            category_counts = df['level3_main_occ'].value_counts().to_dict()
            categories_distribution[os.path.realpath(file_path)] = category_counts

    return categories_distribution


def split_data(df, stratify_col='level3_main_occ', train_size=0.7, val_size=0.15, test_size=0.15):
    train, temp = train_test_split(df, test_size=(val_size + test_size), stratify=df[stratify_col], shuffle=True)
    val, test = train_test_split(temp, test_size=(test_size / (val_size + test_size)), stratify=temp[stratify_col],
                                 shuffle=True)
    return train, val, test


paths = ['./1/1.csv', './2/2.csv', './3/3.csv', './4/4.csv']
for i, file_path in enumerate(paths):
    parent_dir = os.path.split(file_path)[0]
    df = pd.read_csv(file_path)
    train, val, test = split_data(df)
    train.to_csv(os.path.join(parent_dir, f'{i + 1}-train.csv'), index=False)
    val.to_csv(os.path.join(parent_dir, f'{i + 1}-val.csv'), index=False)
    test.to_csv(os.path.join(parent_dir, f'{i + 1}-test.csv'), index=False)
    print(f"Data split for {file_path} completed.")
