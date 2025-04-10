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
from config import TRAIN_TEST_OUTPUT_FILE


def prepare_data_BIO(texts, entity_lists, get_features_for_sentence=None):
    """
    Prepare data for CRF training by extracting features and creating labels
    
    Args:
        texts: List of texts
        entity_lists: List of dictionaries with entity types as keys and lists of entities as values
        get_features_for_sentence: Optional function to extract features from a spaCy doc
        
    Returns:
        X: List of feature dictionaries for each sentence
        y: List of label sequences for each sentence
    """
    X = []
    y = []
    
    nlp = spacy.load('en_core_web_sm')

    for i, text in enumerate(texts):
        doc = nlp(text)
        
        if get_features_for_sentence is not None:
            sentence_features = get_features_for_sentence(doc)
            X.append(sentence_features)
        
        labels = ['O'] * len(doc)
        
        candidate_spans = []
        
        for entity_type, entity_list in entity_lists[i].items():
            if not isinstance(entity_list, list):
                entity_list = [entity_list] if entity_list else []
                
            for entity in entity_list:
                if not entity:
                    continue
                
                entity_lower = entity.lower()
                text_lower = text.lower()
                
                for match in re.finditer(r'\b' + re.escape(entity_lower) + r'\b', text_lower):
                    start_char, end_char = match.span()
                    
                    start_token = None
                    end_token = None
                    
                    for j, token in enumerate(doc):
                        if token.idx <= start_char and token.idx + len(token.text) > start_char:
                            start_token = j
                        if token.idx < end_char and token.idx + len(token.text) >= end_char:
                            end_token = j
                            break
                    
                    if start_token is not None and end_token is not None:
                        span_length = end_token - start_token + 1
                        candidate_spans.append((start_token, end_token + 1, entity_type, span_length))
        
        candidate_spans.sort(key=lambda x: x[3], reverse=True)
        
        final_spans = []
        token_occupancy = set()
        
        for span in candidate_spans:
            start, end, entity_type, _ = span
            span_positions = set(range(start, end))
            if span_positions.intersection(token_occupancy):
                continue
            
            final_spans.append((start, end, entity_type))
            token_occupancy.update(span_positions)
        
        for start, end, entity_type in final_spans:
            if end - start == 1:
                labels[start] = f'B-{entity_type}'
            else:
                labels[start] = f'B-{entity_type}'
                for j in range(start + 1, end):
                    labels[j] = f'I-{entity_type}'
        
        y.append(labels)
    
    return X, y


def calc_hist(df, col_name, threshold, other_bar=False):
    """
    Calculate and visualize the distribution of values in a column.
    
    Args:
        df: DataFrame containing the data
        col_name: Name of the column to analyze
        threshold: Percentage threshold for filtering values
        other_bar: Whether to aggregate values below threshold into an "Other" category
        
    Returns:
        None, displays a plot
    """
    all_conditions = set().union(*df[col_name])
    
    counts = {cond: df[col_name].apply(lambda x: cond in x).sum() for cond in all_conditions}
    
    total_records = len(df)
    percentage = {cond: (counts[cond] / total_records) * 100 for cond in counts}
    
    result_df = pd.DataFrame({
        'Count': counts,
        'Percentage': percentage
    }).sort_values('Count', ascending=False)
    
    above_threshold = result_df[result_df['Percentage'] > threshold].copy()
    below_threshold = result_df[result_df['Percentage'] <= threshold]
    
    other_count = below_threshold['Count'].sum()
    other_percentage = (other_count / total_records) * 100
    
    if other_bar:
        above_threshold.loc['Other'] = [other_count, other_percentage]
    
        above_threshold['LowFreqCount'] = other_count
    
    plt.figure(figsize=(20, 5))
    ax = above_threshold['Percentage'].plot(kind='bar')
    plt.xlabel(col_name)
    plt.ylabel('Percentage (%)')
    plt.title(f'Distribution of Each {col_name} Across Records (in Percentage)')
    plt.show()
    
    plt.show()


def fix_quotes(s):
    """
    Fix common quoting issues in string representation of lists.
    
    Args:
        s: String to fix
        
    Returns:
        Fixed string
    """
    if s.startswith("'\""):
        s = s[1:]
    if s.endswith("''"):
        s = s[:-1]
    return s
    

def convert_conditions(x):
    """
    Convert string representation of a list to an actual list.
    
    Args:
        x: String or list to convert
        
    Returns:
        List object
    """
    if isinstance(x, list):
        return x
    
    x = x.strip()
    x = fix_quotes(x)
    if not (x.startswith('[') and x.endswith(']')):
        x = '[' + x + ']'
    return ast.literal_eval(x)


def convert_string_to_list(df, col_name):
    """
    Convert string representation of lists in a DataFrame column to actual lists.
    
    Args:
        df: DataFrame to modify
        col_name: Name of the column to convert
        
    Returns:
        None, modifies the DataFrame in place
    """
    col_counts = col_name + '_counts'
    for idx, row in df.iterrows():
        count_comma = row[col_name].count(',')
        if count_comma:
            count_comma += 1
        df.at[idx, col_name] = convert_conditions(row[col_name])
        df.at[idx, col_counts] = count_comma


def calc_empty(df, empty_cacl_list):
    """
    Calculate the percentage of empty values in specified columns.
    
    Args:
        df: DataFrame to analyze
        empty_cacl_list: List of column names to check for empty values
        
    Returns:
        None, prints a table with results
    """
    table = [] 
    for header in empty_cacl_list:
        calc = round((len(df[header][df[header].apply(lambda x: len(x)) == 0])/len(df))*100,2)
        table.append([header,calc])

    print(tabulate(table, ['Column', 'Empty percentage'], tablefmt="github"))


def calc_exact_match(df):
    """
    Calculate how well entities match with the text they're extracted from.
    
    Args:
        df: DataFrame containing text and entity columns
        
    Returns:
        None, prints statistics about entity matches
    """
    total_rows = 0
    full_match_count = 0
    match_percentages = []
    missing_examples_count = 3
    
    for idx, row in df.iterrows():
        tot_row_ent = set()
        
        for col_ent in ['Condition', 'Procedure', 'Medication']:
            col_count = col_ent + '_counts'
            tot_row_ent.update(row[col_ent])
        if len(tot_row_ent):
            total_rows += 1
        
            matched_count = 0
            for ent in tot_row_ent:
                if ent.lower() in row['text'].lower():
                    matched_count += 1
        
            if tot_row_ent:
                match_percentage = (matched_count / len(tot_row_ent)) * 100
            else:
                match_percentage = 0
        
            match_percentages.append(match_percentage)
        
            if match_percentage == 100:
                full_match_count += 1
            elif missing_examples_count:
                print('-'*30 + ' ' + str(tot_row_ent) + ' ' + '-'*30)
                print(row['text'])
                print(f"Row match percentage: {match_percentage:.2f}%\n")
                missing_examples_count -= 1
    
    average_match = sum(match_percentages)/len(match_percentages) if match_percentages else 0
    percentage_full_matches = (full_match_count / total_rows) * 100 if total_rows else 0
    
    print("Final Statistics:")
    print(f"Total rows processed: {total_rows}")
    print(f"Rows with 100% match: {full_match_count} ({percentage_full_matches:.2f}%)")
    print(f"Average match percentage across rows: {average_match:.2f}%")
    

def read_train_test_split():
    """
    Read train-test split data from a JSON file.
    
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    with open(TRAIN_TEST_OUTPUT_FILE, 'r') as f:
        data = json.load(f)
    
    X_train = pd.Series(data["X_train"])
    y_train = pd.DataFrame(data["y_train"])
    X_test = pd.Series(data["X_test"])
    y_test = pd.DataFrame(data["y_test"])
    
    return X_train, y_train, X_test, y_test


def calculate_entity_balance(y_train, y_test):
    """
    Calculate balance metrics between training and test sets for different entity types.
    
    Args:
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        Dictionary of balance metrics for each entity type
    """
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
    """
    Create visualizations of entity balance between train and test sets.
    
    Args:
        results: Dictionary of balance metrics from calculate_entity_balance
        
    Returns:
        None, displays plots
    """
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
    """
    Print a formatted table of balance results.
    
    Args:
        balance_results: Dictionary of balance metrics from calculate_entity_balance
        
    Returns:
        None, prints a table
    """
    print("-" * 80)
    print(f"{'Entity Type':<15} {'Balance Score %':<15} {'Overlap %':<15} {'Normallized EMD %':<15} {'KS Stat %':<18}")
    print("-" * 80)
  
    for entity_type, metrics in balance_results.items():
        overlap_info = f"{round(metrics['overlap_ratio']*100,1)}%({metrics['shared_entities']}/{metrics['total_unique']})"
        print(f"{entity_type:<15} {metrics['balance_score']:.1f}%{'':<13} {overlap_info:<17} {round(metrics['earth_movers_distance'],2)*100:.1f}%{'':<10} {round(metrics['ks_statistic'],2)*100:.1f}%")


def plot_entity_distributions(y_train, y_test):
    """
    Plot the distribution of entity types in train and test sets.
    
    Args:
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        None, displays a plot and saves it to a file
    """
    entity_types = ["Condition", "Procedure", "Medication"]    
    train_counts = {}
    test_counts = {}
    
    for entity_type in entity_types:
        train_entities = []
        for entities_list in y_train[entity_type]:
            if isinstance(entities_list, list):
                train_entities.extend(entities_list)
        
        test_entities = []
        for entities_list in y_test[entity_type]:
            if isinstance(entities_list, list):
                test_entities.extend(entities_list)
        
        train_counts[entity_type] = len(train_entities)
        test_counts[entity_type]
