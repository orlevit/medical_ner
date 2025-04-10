# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from config import DATA_FILE, TRAIN_TEST_OUTPUT_FILE
from helper import *
import numpy as np


def preprocess_text(text):
    """
    Preprocesses the input text by performing multiple cleaning operations.
    
    The function:
    1. Converts all text to lowercase
    2. Removes special characters and punctuation by replacing them with spaces
    3. Standardizes whitespace by removing excessive spaces and trimming
    
    Parameters:
        text (str): The input text to be preprocessed
        
    Returns:
        str: The cleaned and standardized text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def create_stratified_split(df, test_size=0.2, random_state=42):
    """
    Creates a stratified data split that ensures balanced distribution of both common 
    and rare medical entities between training and test sets.
    
    The function uses a multi-step approach:
    1. Identifies rare entities (appearing less than 3 times in the dataset)
    2. Segregates texts with rare entities and ensures their representation in training
    3. For remaining texts with common entities, applies clustering for stratification
    4. Combines the splits to create final train and test datasets
    
    Parameters:
        df (DataFrame): Input dataframe containing medical text and entity annotations
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df) containing the split datasets
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    entity_counts = {}
    for col in ['Condition', 'Procedure', 'Medication']:
        for entities in df[col]:
            for entity in entities:
                if entity:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

    rare_entities = {entity for entity, count in entity_counts.items() if count < 3}
    print(f"Number of rare entities (appearing less than 3 times): {len(rare_entities)}")

    contains_rare = []
    for i, row in df.iterrows():
        has_rare = False
        for col in ['Condition', 'Procedure', 'Medication']:
            if any(entity in rare_entities for entity in row[col] if entity):
                has_rare = True
                break
        contains_rare.append(has_rare)
    
    df['contains_rare'] = contains_rare
    rare_texts_df = df[df['contains_rare']].copy()
    common_texts_df = df[~df['contains_rare']].copy()

    print(f"Texts with rare entities: {len(rare_texts_df)} ({len(rare_texts_df)/len(df)*100:.1f}%)")
    print(f"Texts with common entities only: {len(common_texts_df)} ({len(common_texts_df)/len(df)*100:.1f}%)")

    rare_train_df, rare_test_df = train_test_split(rare_texts_df, test_size=0.1, random_state=random_state)

    if len(common_texts_df) > 0:
        vectorizer = TfidfVectorizer(max_features=1000)
        X_tfidf = vectorizer.fit_transform(common_texts_df['processed_text'])
        n_clusters = min(max(5, len(common_texts_df) // 100), 20)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(X_tfidf)
        common_texts_df['cluster'] = clusters
        common_train_df, common_test_df = train_test_split(common_texts_df, test_size=test_size, stratify=common_texts_df['cluster'], random_state=random_state)
    else:
        common_train_df = pd.DataFrame(columns=df.columns)
        common_test_df = pd.DataFrame(columns=df.columns)

    train_df = pd.concat([rare_train_df, common_train_df])
    test_df = pd.concat([rare_test_df, common_test_df])
    
    if 'contains_rare' in train_df.columns:
        train_df = train_df.drop(columns=['contains_rare'])
    if 'cluster' in train_df.columns:
        train_df = train_df.drop(columns=['cluster'])
    if 'contains_rare' in test_df.columns:
        test_df = test_df.drop(columns=['contains_rare'])
    if 'cluster' in test_df.columns:
        test_df = test_df.drop(columns=['cluster'])
    
    return train_df, test_df


def train_test_splitting():
    """
    Main function that performs the train-test split for medical entity extraction.
    
    This function:
    1. Loads the medical text dataset from a configured file path
    2. Preprocesses the data including string-to-list conversion for entity fields
    3. Analyzes entity distributions and cardinality
    4. Creates a stratified train-test split that preserves entity distributions
    5. Saves the resulting split data to the configured output file
    
    The function preserves both features (processed text) and labels (medical entities)
    in the output, formatted as a JSON file for downstream model training.
    
    Returns:
        None: The function writes results to a JSON file but doesn't return values
    """
    if not os.path.exists(TRAIN_TEST_OUTPUT_FILE):
        df = pd.read_csv(DATA_FILE)
        df.fillna(value='', inplace=True)
        
        convert_string_to_list(df, 'Procedure')
        convert_string_to_list(df, 'Condition')
        convert_string_to_list(df, 'Medication')
        
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        print("\nAnalyzing entity cardinality:")
        all_conditions = [item for sublist in df['Condition'].tolist() for item in sublist if item]
        all_procedures = [item for sublist in df['Procedure'].tolist() for item in sublist if item]
        all_medications = [item for sublist in df['Medication'].tolist() for item in sublist if item]
        
        unique_conditions = set(all_conditions)
        unique_procedures = set(all_procedures)
        unique_medications = set(all_medications)
        
        print(f"Data rows: {len(df)}")
        print(f"Unique Conditions: {len(unique_conditions)}")
        print(f"Unique Procedures: {len(unique_procedures)}")
        print(f"Unique Medications: {len(unique_medications)}")
        
        train_df, test_df = create_stratified_split(df, test_size=0.2, random_state=42)
            
        X_train = train_df['processed_text']
        y_train = train_df[['Condition', 'Procedure', 'Medication']]
        X_test = test_df['processed_text']
        y_test = test_df[['Condition', 'Procedure', 'Medication']]

        data = {
            "X_train": X_train.tolist(),
            "y_train": y_train.to_dict(orient="records"),
            "X_test": X_test.tolist(),
            "y_test": y_test.to_dict(orient="records")
        }
        
        with open(TRAIN_TEST_OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)
  
# %%
## Uncomment to run the model
# train_test_splitting()
# %%

