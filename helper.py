import re
import ast
import json
import spacy
import pandas as pd
from os import wait
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import Counter
from config import  TRAIN_TEST_OUTPUT_FILE




def prepare_data_BIO(texts, entity_lists, get_features_for_sentence=None):
    """
    Prepare data for CRF training by extracting features and creating labels
    
    Args:
        texts: List of texts
        entity_lists: List of dictionaries with entity types as keys and lists of entities as values
        
    Returns:
        X: List of feature dictionaries for each sentence
        y: List of label sequences for each sentence
    """
    X = []  # Features
    y = []  # Labels
    
    nlp = spacy.load('en_core_web_sm')

    for i, text in enumerate(texts):
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract features for each token
        if get_features_for_sentence is not None:
            sentence_features = get_features_for_sentence(doc)
            X.append(sentence_features)
        
        # Create labels for each token
        labels = ['O'] * len(doc)  # Default label is 'O' (Outside)
        
        # Collect all entity spans before assigning labels
        candidate_spans = []
        
        # Find entities in the text and collect their spans
        for entity_type, entity_list in entity_lists[i].items():
            if not isinstance(entity_list, list):
                # Convert to list if it's not already
                entity_list = [entity_list] if entity_list else []
                
            for entity in entity_list:
                if not entity:
                    continue
                
                entity_lower = entity.lower()
                text_lower = text.lower()
                
                # Find all occurrences of the entity
                for match in re.finditer(r'\b' + re.escape(entity_lower) + r'\b', text_lower):
                    start_char, end_char = match.span()
                    
                    # Find token indices that correspond to this entity
                    start_token = None
                    end_token = None
                    
                    for j, token in enumerate(doc):
                        if token.idx <= start_char and token.idx + len(token.text) > start_char:
                            start_token = j
                        if token.idx < end_char and token.idx + len(token.text) >= end_char:
                            end_token = j
                            break
                    
                    if start_token is not None and end_token is not None:
                        # Add to candidate spans with span length as score (for sorting)
                        span_length = end_token - start_token + 1
                        candidate_spans.append((start_token, end_token + 1, entity_type, span_length))
        
        # Sort spans by length in descending order (prefer longer spans)
        candidate_spans.sort(key=lambda x: x[3], reverse=True)
        
        # Filter out overlapping spans
        final_spans = []
        token_occupancy = set()
        
        for span in candidate_spans:
            start, end, entity_type, _ = span
            span_positions = set(range(start, end))
            if span_positions.intersection(token_occupancy):
                # Skip if any token in this span is already part of another span
                continue
            
            final_spans.append((start, end, entity_type))
            token_occupancy.update(span_positions)
        
        # Assign labels based on final spans
        for start, end, entity_type in final_spans:
            # Single-token entity
            if end - start == 1:
                labels[start] = f'B-{entity_type}'
            # Multi-token entity
            else:
                labels[start] = f'B-{entity_type}'  # Beginning
                for j in range(start + 1, end):
                    labels[j] = f'I-{entity_type}'  # Inside
        
        y.append(labels)
    
    return X, y

def calc_hist(df, col_name, threshold, other_bar=False):
    # Get all unique conditions across rows.
    all_conditions = set().union(*df[col_name])
    
    # Compute the count of each condition.
    counts = {cond: df[col_name].apply(lambda x: cond in x).sum() for cond in all_conditions}
    
    total_records = len(df)
    # Calculate percentage appearance of each condition.
    percentage = {cond: (counts[cond] / total_records) * 100 for cond in counts}
    
    # Build a DataFrame from counts and percentage dictionaries.
    result_df = pd.DataFrame({
        'Count': counts,
        'Percentage': percentage
    }).sort_values('Count', ascending=False)
    
    # Separate conditions above and below the threshold.
    above_threshold = result_df[result_df['Percentage'] > threshold].copy()
    below_threshold = result_df[result_df['Percentage'] <= threshold]
    
    # Aggregate all low-frequency counts.
    other_count = below_threshold['Count'].sum()
    other_percentage = (other_count / total_records) * 100
    
    # Add a new row for the aggregated "Other" category.
    if other_bar:
        above_threshold.loc['Other'] = [other_count, other_percentage]
    
        # Add a new column showing the total low-frequency count (same value for every row).
        above_threshold['LowFreqCount'] = other_count
    
    # Plot the bar chart for Percentage values.
    plt.figure(figsize=(20, 5))
    ax = above_threshold['Percentage'].plot(kind='bar')
    plt.xlabel(col_name)
    plt.ylabel('Percentage (%)')
    plt.title(f'Distribution of Each {col_name} Across Records (in Percentage)')
    plt.show()
    
    plt.show()

def fix_quotes(s):
    # If the string starts with a single quote followed by a double quote, remove the leading single quote
    if s.startswith("'\""):
        s = s[1:]
    # If the string ends with two single quotes, remove the last extra quote
    if s.endswith("''"):
        s = s[:-1]
    return s
    
def convert_conditions(x):

    if isinstance(x, list):
        return x
    
    x = x.strip()
    x = fix_quotes(x)
    # If the string does not start with '[' and end with ']', add them
    if not (x.startswith('[') and x.endswith(']')):
        x = '[' + x + ']'
        # Safely evaluate the string to a Python literal
    return ast.literal_eval(x)

def convert_string_to_list(df, col_name):
    col_counts  = col_name + '_counts'
    for idx, row in df.iterrows():
        count_comma = row[col_name].count(',')
        if count_comma:
            count_comma += 1
        df.at[idx, col_name] = convert_conditions(row[col_name])
        df.at[idx, col_counts] = count_comma

def calc_empty(df, empty_cacl_list):

    table = [] 
    for header in empty_cacl_list:
        calc = round((len(df[header][df[header].apply(lambda x: len(x)) == 0])/len(df))*100,2)
        table.append([header,calc])

    print(tabulate(table, ['Column', 'Empty percentage'], tablefmt="github"))


def calc_exact_match(df):
    total_rows = 0
    full_match_count = 0  # Count of rows where all entities are found
    match_percentages = []  # List of percentages for each row
    missing_examples_count = 3
    
    for idx, row in df.iterrows():
        tot_row_ent = set()
        
        # Build tot_row_ent from lists in the columns:
        for col_ent in ['Condition', 'Procedure', 'Medication']:
            col_count = col_ent + '_counts'
            # Assuming row[col_ent] is a list of strings,
            # and that even if row[col_count] is false, we still want these entities.
            tot_row_ent.update(row[col_ent])
        if len(tot_row_ent):
            # Count rows processed
            total_rows += 1
        
            # For each entity in the tot_row_ent check if it appears in the text
            # (using a simple case-insensitive "in" check)
            matched_count = 0
            for ent in tot_row_ent:
                if ent.lower() in row['text'].lower():
                    matched_count += 1
        
            # Calculate match percentage for this row
            if tot_row_ent:
                match_percentage = (matched_count / len(tot_row_ent)) * 100
            else:
                match_percentage = 0  # Or handle rows with no entities separately
        
            match_percentages.append(match_percentage)
        
            # If 100% of the entities are found in the text, count this as a full match
            if match_percentage == 100:
                full_match_count += 1
            elif missing_examples_count:
                # (Optional) Print each row's details
                print('-'*30 + ' ' + str(tot_row_ent) + ' ' + '-'*30)
                print(row['text'])
                print(f"Row match percentage: {match_percentage:.2f}%\n")
                missing_examples_count -= 1
    # Compute final statistics
    average_match = sum(match_percentages)/len(match_percentages) if match_percentages else 0
    percentage_full_matches = (full_match_count / total_rows) * 100 if total_rows else 0
    
    print("Final Statistics:")
    print(f"Total rows processed: {total_rows}")
    print(f"Rows with 100% match: {full_match_count} ({percentage_full_matches:.2f}%)")
    print(f"Average match percentage across rows: {average_match:.2f}%")
    


def read_train_test_split():
    with open(TRAIN_TEST_OUTPUT_FILE, 'r') as f:
        data = json.load(f)
    
    X_train = pd.Series(data["X_train"])
    y_train = pd.DataFrame(data["y_train"])
    X_test = pd.Series(data["X_test"])
    y_test = pd.DataFrame(data["y_test"])
    
    return X_train, y_train, X_test, y_test

import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import wasserstein_distance, ks_2samp
import matplotlib.pyplot as plt

def calculate_entity_balance(y_train, y_test):
    entity_types = ["Condition", "Procedure", "Medication"]
    results = {}
    
    for entity_type in entity_types:
        
        train_entities = []
        for entities_list in y_train[entity_type]:
            if isinstance(entities_list, list):
                train_entities.extend(entities_list)
                
        test_entities = []
        for entities_list in y_test[entity_type]:
            if isinstance(entities_list, list):
                test_entities.extend(entities_list)
        
        train_counter = Counter(train_entities)
        test_counter = Counter(test_entities)
        
        all_entities = set(train_counter.keys()) | set(test_counter.keys())
        
        train_total = len(train_entities)
        test_total = len(test_entities)
        
        train_dist = {entity: train_counter.get(entity, 0) / train_total 
                      for entity in all_entities}
        test_dist = {entity: test_counter.get(entity, 0) / test_total 
                     for entity in all_entities}
        
        train_coverage = len(train_counter) / len(all_entities)
        test_coverage = len(test_counter) / len(all_entities)
        shared_entities = set(train_counter.keys()) & set(test_counter.keys())
        overlap_ratio = len(shared_entities) / len(all_entities)
        
        all_entities_list = list(all_entities)
        train_values = [train_dist.get(entity, 0) for entity in all_entities_list]
        test_values = [test_dist.get(entity, 0) for entity in all_entities_list]
        
        emd = wasserstein_distance(train_values, test_values)
        
        ks_stat, ks_pvalue = ks_2samp(train_values, test_values)
        
        train_array = np.array(train_values)
        test_array = np.array(test_values)
        m = 0.5 * (train_array + test_array)
        
        def safe_log(x, base=np.e):
            return np.log(x, out=np.zeros_like(x), where=(x!=0))
            
        def kl_divergence(p, q):
            non_zero = p > 0
            safe_q = np.copy(q)
            safe_q[non_zero & (safe_q == 0)] = 1e-10
            return np.sum(p[non_zero] * safe_log(p[non_zero] / safe_q[non_zero]))
            
        js_div = 0.5 * kl_divergence(train_array, m) + 0.5 * kl_divergence(test_array, m)
        
        normalized_emd = 1 - min(emd, 1)
        normalized_ks = 1 - ks_stat
        normalized_js = 1 - min(js_div, 1)
        
        balance_score = (normalized_emd * 0.3 + 
                         normalized_ks * 0.1 + 
                         normalized_js * 0.1 + 
                         overlap_ratio * 0.4) * 100
        
        results[entity_type] = {
            "unique_in_train": len(train_counter),
            "unique_in_test": len(test_counter),
            "total_unique": len(all_entities),
            "shared_entities": len(shared_entities),
            "overlap_ratio": overlap_ratio,
            "earth_movers_distance": emd,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "jensen_shannon_div": js_div,
            "balance_score": balance_score
        }
        
    return results

def create_balance_visualization(results):
    entity_types = list(results.keys())
    balance_scores = [results[et]["balance_score"] for et in entity_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(entity_types, balance_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom')
    
    plt.title('Entity Type Balance Score (Train vs Test)')
    plt.ylabel('Balance Score (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
    
    overlap_ratios = [results[et]["overlap_ratio"] * 100 for et in entity_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(entity_types, overlap_ratios, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('Entity Overlap Between Train and Test Sets')
    plt.ylabel('Overlap Percentage (%)')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def print_balance_results(balance_results):
    print("-" * 80)
    print(f"{'Entity Type':<15} {'Balance Score %':<15} {'Overlap %':<15} {'Normallized EMD %':<15} {'KS Stat %':<18}")
    print("-" * 80)
  
    for entity_type, metrics in balance_results.items():
        overlap_info = f"{round(metrics['overlap_ratio']*100,1)}%({metrics['shared_entities']}/{metrics['total_unique']})"
        # Don't convert balance_score to string yet
        print(f"{entity_type:<15} {metrics['balance_score']:.1f}%{'':<13} {overlap_info:<17} {round(metrics['earth_movers_distance'],2)*100:.1f}%{'':<10} {round(metrics['ks_statistic'],2)*100:.1f}%")

def plot_entity_distributions(y_train, y_test):
    entity_types = ["Condition", "Procedure", "Medication"]    
    # Get raw counts for both datasets
    train_counts = {}
    test_counts = {}
    
    for entity_type in entity_types:
        # Get all entities from train set
        train_entities = []
        for entities_list in y_train[entity_type]:
            if isinstance(entities_list, list):
                train_entities.extend(entities_list)
        
        # Get all entities from test set
        test_entities = []
        for entities_list in y_test[entity_type]:
            if isinstance(entities_list, list):
                test_entities.extend(entities_list)
        
        # Store raw counts
        train_counts[entity_type] = len(train_entities)
        test_counts[entity_type] = len(test_entities)
    
    # Create normalized counts (percentages)
    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    
    train_pct = {k: v/train_total*100 for k, v in train_counts.items()}
    test_pct = {k: v/test_total*100 for k, v in test_counts.items()}
    
    # Plot the distribution
    x = np.arange(len(entity_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    train_bars = ax.bar(x - width/2, [train_pct[et] for et in entity_types], width, label='Train', color='#3498db')
    test_bars = ax.bar(x + width/2, [test_pct[et] for et in entity_types], width, label='Test', color='#e74c3c')
    
    # Add counts on top of the bars
    def add_labels(bars, counts):
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}\n({height:.1f}%)',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, -2),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')
    
    add_labels(train_bars, [train_counts[et] for et in entity_types])
    add_labels(test_bars, [test_counts[et] for et in entity_types])
    
    # Add labels and title
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Distribution of Entity Types in Train vs Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(entity_types)
    ax.legend()
    
    # Add a second y-axis for raw counts
    ax2 = ax.twinx()
    max_count = max(max(train_counts.values()), max(test_counts.values()))
    ax2.set_ylim(0, max_count * 1.15)  # Match the percentage axis but with counts
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('entity_type_distribution.png')
    plt.show()
