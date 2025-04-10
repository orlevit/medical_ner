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
from helper import *
from config import OUTPUT_DIR, RULE_BASE_MODEL_OUTPUT_FILE
import pandas as pd

def create_entity_dictionaries(df):
    """
    Creates dictionaries of unique entities (conditions, procedures, and medications) from a dataframe.
    
    Args:
        df: DataFrame containing 'Condition', 'Procedure', and 'Medication' columns with list values
        
    Returns:
        Dictionary mapping entity types to sets of unique entity values
    """
    conditions = set()
    procedures = set()
    medications = set()
    for _, row in df.iterrows():
        conditions.update(row['Condition'])
        procedures.update(row['Procedure'])
        medications.update(row['Medication'])
    conditions.discard('')
    procedures.discard('')
    medications.discard('')
    return {'Condition': conditions, 'Procedure': procedures, 'Medication': medications}

def rule_based_ner(text, entity_dicts):
    """
    Performs rule-based named entity recognition on text using dictionaries of entities.
    
    Args:
        text: Input text to analyze
        entity_dicts: Dictionary of entity types mapped to sets of entity values
        
    Returns:
        Dictionary of entity types mapped to lists of found entities in the text
    """
    text_lower = text.lower()
    found_entities = {'Condition': [], 'Procedure': [], 'Medication': []}
    for entity_type, entities in entity_dicts.items():
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in text_lower:
                found_entities[entity_type].append(entity)
    return found_entities

def evaluate_rule_based_ner(texts, true_entities, entity_dicts):
    """
    Evaluates the performance of rule-based NER by comparing predictions to ground truth.
    
    Args:
        texts: List of input texts
        true_entities: List of dictionaries containing ground truth entities for each text
        entity_dicts: Dictionary of entity types mapped to sets of entity values
        
    Returns:
        Dictionary with performance metrics (precision, recall, F1) overall and per entity type
    """
    results = {
        'overall': {'tp': 0, 'fp': 0, 'fn': 0},
        'Condition': {'tp': 0, 'fp': 0, 'fn': 0},
        'Procedure': {'tp': 0, 'fp': 0, 'fn': 0},
        'Medication': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    for i, text in enumerate(texts):
        pred_entities = rule_based_ner(text, entity_dicts)
        for entity_type in ['Condition', 'Procedure', 'Medication']:
            true_set = set(true_entities[i][entity_type])
            pred_set = set(pred_entities[entity_type])
            tp = len(true_set.intersection(pred_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            results[entity_type]['tp'] += tp
            results[entity_type]['fp'] += fp
            results[entity_type]['fn'] += fn
            results['overall']['tp'] += tp
            results['overall']['fp'] += fp
            results['overall']['fn'] += fn
    for key, counts in results.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results[key]['precision'] = precision
        results[key]['recall'] = recall
        results[key]['f1'] = f1
    return results

def generate_rule_based_output(texts, entity_dicts):
    """
    Generates output dataframe with predicted entities for each input text.
    
    Args:
        texts: List of input texts
        entity_dicts: Dictionary of entity types mapped to sets of entity values
        
    Returns:
        DataFrame with columns for text and identified entities (conditions, procedures, medications)
    """
    results = []
    for text in texts:
        pred_entities = rule_based_ner(text, entity_dicts)
        condition_str = ', '.join(set(pred_entities['Condition'])) if pred_entities['Condition'] else ''
        procedure_str = ', '.join(set(pred_entities['Procedure'])) if pred_entities['Procedure'] else ''
        medication_str = ', '.join(set(pred_entities['Medication'])) if pred_entities['Medication'] else ''
        results.append({
            'text': text,
            'Condition': condition_str,
            'Procedure': procedure_str,
            'Medication': medication_str
        })
    return pd.DataFrame(results)

def rule_base_main():
    """
    Main function to run the rule-based NER system:
    1. Loads training and test data
    2. Creates entity dictionaries from training data
    3. Evaluates the rule-based NER on test data
    4. Prints performance metrics
    5. Generates and saves output predictions to a CSV file
    """
    X_train, y_train, X_test, y_test = read_train_test_split()
    train_df = pd.DataFrame({
        'processed_text': X_train,
        'Condition': y_train['Condition'],
        'Procedure': y_train['Procedure'],
        'Medication': y_train['Medication']
    })
    entity_dictionaries = create_entity_dictionaries(train_df)
    test_entities = []
    for i in range(len(y_test)):
        test_entities.append({
            "Condition": y_test.iloc[i]['Condition'],
            "Procedure": y_test.iloc[i]['Procedure'],
            "Medication": y_test.iloc[i]['Medication']
        })
    rb_metrics = evaluate_rule_based_ner(X_test.tolist(), test_entities, entity_dictionaries)
    print("\nRule-Based NER Performance:")
    print(f"Overall Precision: {rb_metrics['overall']['precision']:.4f}")
    print(f"Overall Recall: {rb_metrics['overall']['recall']:.4f}")
    print(f"Overall F1 Score: {rb_metrics['overall']['f1']:.4f}")
    print("\nPerformance by Entity Type:")
    for entity_type in ['Condition', 'Procedure', 'Medication']:
        print(f"\n{entity_type}:")
        print(f"Precision: {rb_metrics[entity_type]['precision']:.4f}")
        print(f"Recall: {rb_metrics[entity_type]['recall']:.4f}")
        print(f"F1 Score: {rb_metrics[entity_type]['f1']:.4f}")
    output_df = generate_rule_based_output(X_test.tolist(), entity_dictionaries)
    output_df.to_csv(RULE_BASE_MODEL_OUTPUT_FILE, index=False)
# %%
## Uncomment to run the model
# rule_base_main()
# %%
