# ---
# jupyter:
#   jupytext:
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
# %load_ext autoreload
# %autoreload 2

# %%
# todo:
#spacy ner  model resourc - https://spacy.io/universe/project/video-spacys-ner-model
# all condition match
# medspacy - designated , unclear dataset traeind: RESORCE - https://github.com/medspacy/medspacy

# %%
import sys
import os
import pandas as pd
import seaborn as sns
import  numpy as np
from train_test_split import train_test_splitting
from config import DATA_FILE, EMPTY_HIST_THRESHOLD_PROCEDURE, EMPTY_HIST_THRESHOLD_CONDITION, EMPTY_HIST_THRESHOLD_MEDICATION
from helper import *

df = pd.read_csv(DATA_FILE)
df.fillna(value='', inplace=True)

convert_string_to_list(df, 'Procedure')
convert_string_to_list(df, 'Condition')
convert_string_to_list(df, 'Medication')

empty_cacl_list = ['Condition', 'Procedure', 'Medication']
uniqe_Conditions = set().union(*df['Condition'])
uniqe_Procedures = set().union(*df['Procedure'])
uniqe_Medication = set().union(*df['Medication'])

def is_non_empty_list(x):
    return isinstance(x, list) and len(x) > 0

# Apply the function to the specified columns and create a mask
columns_of_interest = ['Condition', 'Procedure', 'Medication']
mask = df[columns_of_interest].applymap(is_non_empty_list)

# Select rows where all specified columns contain non-empty lists
df_non_empty = df[mask.any(axis=1)]

print(f'Lengths: Data : {len(df)}, None empty rows: {len(df_non_empty)}\n',
f'Uniqe Conditions: {len(uniqe_Conditions)}, uniqe_Procedures: {len(uniqe_Procedures)}, uniqe Medication:{len(uniqe_Medication)}')
calc_empty(df, empty_cacl_list)


# %%
def is_non_empty_list(x):
    return isinstance(x, list) and len(x) > 0

# Apply the function to the specified columns and create a mask
columns_of_interest = ['Condition', 'Procedure', 'Medication']
mask = df[columns_of_interest].applymap(is_non_empty_list)

# Select rows where all specified columns contain non-empty lists
df_non_empty = df[mask.any(axis=1)]
len(df_non_empty)

# %%
calc_hist(df, 'Procedure', EMPTY_HIST_THRESHOLD_PROCEDURE,other_bar=True)
# %%
calc_hist(df, 'Condition', EMPTY_HIST_THRESHOLD_CONDITION, other_bar=False)
# %%
calc_hist(df, 'Medication', EMPTY_HIST_THRESHOLD_MEDICATION,other_bar=True)
# %% [markdown]
# # Text

# %%
# Text length analysis
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(12, 6))
sns.histplot(df['text_length'], bins=30)
plt.title('Distribution of Text Lengths')
plt.xlabel('Length (characters)')
plt.ylabel('Frequency')
plt.show()

print(f"\nAverage text length: {df['text_length'].mean():.2f} characters")
print(f"Median text length: {df['text_length'].median()} characters")
print(f"Min text length: {df['text_length'].min()} characters")
print(f"Max text length: {df['text_length'].max()} characters")

# %%
import pandas as pd
import numpy as np

def analyze_text_length_by_entity(df):
    # First calculate text length
    df['text_length'] = df['text'].apply(len)
    
    # Overall text statistics
    print("OVERALL TEXT STATISTICS")
    print(f"Average text length: {df['text_length'].mean():.2f} characters")
    print(f"Median text length: {df['text_length'].median()} characters")
    print(f"Min text length: {df['text_length'].min()} characters")
    print(f"Max text length: {df['text_length'].max()} characters")
    
    # Create a summary dataframe to store statistics by entity type
    entity_types = ["Condition", "Procedure", "Medication"]
    summary_data = []
    
    # For each entity type, calculate metrics
    for entity_type in entity_types:
        # Find texts that have this entity type
        texts_with_entity = df[df[entity_type].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        
        if len(texts_with_entity) > 0:
            avg_length = texts_with_entity['text_length'].mean()
            median_length = texts_with_entity['text_length'].median()
            min_length = texts_with_entity['text_length'].min()
            max_length = texts_with_entity['text_length'].max()
            count = len(texts_with_entity)
            
            # For texts with this entity, what's the average entity count?
            avg_entity_count = texts_with_entity[entity_type].apply(len).mean()
            
            # Calculate density: entities per 100 characters
            entity_density = texts_with_entity[entity_type].apply(len).sum() / texts_with_entity['text_length'].sum() * 100
            
            summary_data.append({
                'Entity Type': entity_type,
                'Count': count,
                'Avg Text Length': round(avg_length, 2),
                'Median Text Length': int(median_length),
                'Min Text Length': min_length,
                'Max Text Length': max_length,
                'Avg Entities per Text': round(avg_entity_count, 2),
                'Entity Density (per 100 chars)': round(entity_density, 2)
            })
        else:
            summary_data.append({
                'Entity Type': entity_type,
                'Count': 0,
                'Avg Text Length': 0,
                'Median Text Length': 0,
                'Min Text Length': 0,
                'Max Text Length': 0,
                'Avg Entities per Text': 0,
                'Entity Density (per 100 chars)': 0
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Print formatted summary
    print("\nTEXT STATISTICS BY ENTITY TYPE")
    print(summary_df.to_string(index=False))
    
    # Additional analysis: text with multiple entity types
    print("\nTEXTS WITH MULTIPLE ENTITY TYPES")
    
    # Count how many entity types each text has
    def count_entity_types(row):
        count = 0
        for entity_type in entity_types:
            if isinstance(row[entity_type], list) and len(row[entity_type]) > 0:
                count += 1
        return count
    
    df['entity_type_count'] = df.apply(count_entity_types, axis=1)
    
    # Group by count and calculate statistics
    entity_count_stats = df.groupby('entity_type_count').agg({
        'text_length': ['count', 'mean', 'median', 'min', 'max']
    })
    
    # Rename columns for better readability
    entity_count_stats.columns = ['Count', 'Avg Length', 'Median Length', 'Min Length', 'Max Length']
    entity_count_stats = entity_count_stats.reset_index()
    entity_count_stats = entity_count_stats.rename(columns={'entity_type_count': 'Number of Entity Types'})
    
    # Format the values
    entity_count_stats['Avg Length'] = entity_count_stats['Avg Length'].round(2)
    entity_count_stats['Median Length'] = entity_count_stats['Median Length'].astype(int)
    
    print(entity_count_stats.to_string(index=False))
    
    return summary_df, entity_count_stats

# Example usage:
summary, multientity_stats = analyze_text_length_by_entity(df)

# %%
for idx, row in df.iterrows():
    tot_row_ent = set()
    
    for col_ent in ['Condition', 'Procedure', 'Medication']:
        col_count = col_ent + '_counts'

        if row[col_count]:
            tot_row_ent.update(row[col_ent])  # flatten list into the set
        else:
            tot_row_ent.update(row[col_ent])  # do the same even if it's zero (can skip if needed)

    print('-'*30 +' ' + str(tot_row_ent) +' ' + '-'*30)
    print(row['text'])
    print('\n')
    if idx == 5:
        break


# %% [markdown]
# # Check if the exact words match in the text

# %%
calc_exact_match(df)

# %%
uniqe_phrases = set().union(*df['Condition']).union(*df['Procedure']).union(*df['Medication'])

matches = []

# Compare each phrase with all the others
for i, phrase in enumerate(uniqe_phrases):
    for j, other_phrase in enumerate(uniqe_phrases):
        if i != j:  # Ensure we don't compare the phrase with itself
            # Check using lower() to perform a case-insensitive search.
            if phrase.lower() in other_phrase.lower():
                matches.append((phrase, other_phrase))
                
print(f'percentage of phrase inside another phrase: {len(matches)}/{len(uniqe_phrases)} = {round(len(matches)/len(uniqe_phrases),2)}')

# %%
counts = 0
for idx, row in df.iterrows():
    tot_row_ent = set()
    
    for col_ent in ['Condition', 'Procedure', 'Medication']:
        col_count = col_ent + '_counts'

        if row[col_count]:
            tot_row_ent.update(row[col_ent])  # flatten list into the set
        else:
            tot_row_ent.update(row[col_ent])  # do the same even if it's zero (can skip if needed)

    if 'Bone Cancer' in row['text']:
        print('-'*30 +' ' + str(tot_row_ent) +' ' + '-'*30)
        print(row['text'])
        print('\n')
        counts += 1
        if counts == 1:
            break

# %%
df2 = df.copy()


def substring_entity_analysis(row):
    # Combine phrases from the three columns.
    phrases = []
    for col in ['Condition', 'Procedure', 'Medication']:
        cell = row[col]
        if isinstance(cell, list):
            phrases.extend(cell)
        elif pd.notnull(cell):
            phrases.append(cell)
    
    # Remove duplicates.
    phrases = list(set(phrases))
    
    # Count how many entities (phrases) are a substring of another
    substring_entity_count = 0
    for i in range(len(phrases)):
        current_phrase = phrases[i].lower()
        # Check if this phrase is contained in any other phrase.
        for j in range(len(phrases)):
            if i != j:
                if current_phrase in phrases[j].lower():
                    substring_entity_count += 1
                    break  # Stop after the first match
    total_entities = len(phrases)
    percent_entity = (substring_entity_count / total_entities * 100) if total_entities > 0 else 0

    return pd.Series({
        'substring_entities_count': substring_entity_count,
        'total_entities': total_entities,
        'percent_substring_entities': percent_entity,
        'row_has_substring': substring_entity_count > 0
    })

# Apply the per-row analysis.
results = df2.apply(substring_entity_analysis, axis=1)
df2 = pd.concat([df2, results], axis=1)

# Compute overall statistics:
# 1. Percentage of rows that contain at least one substring entity.
percentage_rows = df2['row_has_substring'].mean() * 100

# 2. Overall percentage of substring entities among all entities.
total_substring_entities = df2['substring_entities_count'].sum()
total_entities = df2['total_entities'].sum()
overall_percent_entities = (total_substring_entities / total_entities * 100) if total_entities > 0 else 0

# Output results.
print("Number of rows:", len(df2))
print(f"Rows with at least one substring entity: {df2['row_has_substring'].sum()} ({percentage_rows:.2f}%)")
print(f"Overall percentage of substring entities among all entities: {overall_percent_entities:.2f}%")

# %%
df.iloc[23:28]

# %%
train_test_splitting()
X_train, y_train, X_test, y_test = read_train_test_split()


# %%
balance_results = calculate_entity_balance(y_train, y_test)
print_balance_results(balance_results)

# %%
create_balance_visualization(balance_results)

# %%
plot_entity_distributions(y_train, y_test)

# %%
