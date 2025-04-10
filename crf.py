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
import pandas as pd
import spacy
import re
import warnings
from sklearn.metrics import precision_recall_fscore_support
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from helper import read_train_test_split, prepare_data_BIO
from config import CRF_MODEL_OUTPUT_FILE
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_sm')

def evaluate_crf_model(crf_model, X_test, y_test):
    """
    Evaluates a trained CRF model using test data and calculates performance metrics.
    
    Args:
        crf_model: Trained CRF model
        X_test: Test features
        y_test: Test labels in BIO format
    
    Returns:
        results: Dictionary containing precision, recall, and F1 scores for each entity type and overall
        report: Classification report as string
        y_pred: Predicted labels
    """
    y_pred = crf_model.predict(X_test)
    
    y_true_flat = [label for sublist in y_test for label in sublist]
    y_pred_flat = [label for sublist in y_pred for label in sublist]
    
    labels = set(y_true_flat) - {'O'}
    sorted_labels = sorted(list(labels), key=lambda name: (name[1:], name[0]))
    
    report = flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=4)
    
    results = {}
    
    entity_types = set()
    for label in sorted_labels:
        entity_type = label[2:]
        entity_types.add(entity_type)
    
    for entity_type in entity_types:
        entity_labels = [label for label in sorted_labels if label.endswith(entity_type)]
        
        y_true_entity = ['1' if label.endswith(entity_type) else '0' for label in y_true_flat]
        y_pred_entity = ['1' if label.endswith(entity_type) else '0' for label in y_pred_flat]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_entity, y_pred_entity, average='binary', pos_label='1'
        )
        
        results[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_flat, y_pred_flat, average='micro', labels=sorted_labels
    )
    
    results['overall'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return results, report, y_pred

def get_features_for_sentence(sentence):
    """
    Extracts features for all words in a sentence.
    
    Args:
        sentence: List of spaCy tokens representing a sentence
    
    Returns:
        List of feature dictionaries for each word in the sentence
    """
    return [get_features_for_one_word(i, sentence) for i in range(len(sentence))]

def get_features_for_one_word(cur_loc, sentence):
    """
    Extracts features for a single word in a sentence.
    
    Features include word text, POS tag, dependency relation, head word,
    suffix, capitalization, and contextual features from surrounding words.
    
    Args:
        cur_loc: Index of the current word
        sentence: List of spaCy tokens representing a sentence
    
    Returns:
        List of features for the word
    """
    end_loc = len(sentence) - 1
    word = sentence[cur_loc]

    word_text = word.text if hasattr(word, 'text') else word.orth_
    word_pos = word.pos_ if hasattr(word, 'pos_') else 'NONE'
    word_dep = word.dep_ if hasattr(word, 'dep_') else 'NONE'
    
    try:
        head_text = word.head.text if hasattr(word.head, 'text') else word.head.orth_
    except:
        head_text = 'NONE'
    
    if len(word_text) >= 3:
        last_three = word_text[-3:]
    else:
        last_three = word_text
        
    try:
        starts_with_capital = word_text[0].isupper() if word_text else False
    except:
        starts_with_capital = False

    features = [
        f'word{0}.lower=' + word_text.lower(),
        f'word{0}.postag=' + word_pos,
        f'word{0}[-3:]=' + last_three,
        f'word{0}.dep=' + word_dep,
        f'word{0}.head=' + head_text,
        f'word{0}.isupper={word_text.isupper()}',
        f'word{0}.isdigit={word_text.isdigit()}',
        f'word{0}.startsWithCapital={starts_with_capital}'
    ]
    
    if cur_loc > 0:
        prev_word = sentence[cur_loc - 1]
        prev_word_text = prev_word.text if hasattr(prev_word, 'text') else prev_word.orth_
        prev_word_pos = prev_word.pos_ if hasattr(prev_word, 'pos_') else 'NONE'
        prev_word_dep = prev_word.dep_ if hasattr(prev_word, 'dep_') else 'NONE'
        
        try:
            prev_head_text = prev_word.head.text if hasattr(prev_word.head, 'text') else prev_word.head.orth_
        except:
            prev_head_text = 'NONE'
        
        if len(prev_word_text) >= 3:
            prev_last_three = prev_word_text[-3:]
        else:
            prev_last_three = prev_word_text
            
        try:
            prev_starts_with_capital = prev_word_text[0].isupper() if prev_word_text else False
        except:
            prev_starts_with_capital = False
        
        features.extend([
            f'word{-1}.lower=' + prev_word_text.lower(),
            f'word{-1}.postag=' + prev_word_pos,
            f'word{-1}[-3:]=' + prev_last_three,
            f'word{-1}.dep=' + prev_word_dep,
            f'word{-1}.head=' + prev_head_text,
            f'word{-1}.isupper={prev_word_text.isupper()}',
            f'word{-1}.isdigit={prev_word_text.isdigit()}',
            f'word{-1}.startsWithCapital={prev_starts_with_capital}'
        ])
    else:
        features.append('BEG')

    if cur_loc < end_loc:
        next_word = sentence[cur_loc + 1]
        next_word_text = next_word.text if hasattr(next_word, 'text') else next_word.orth_
        next_word_pos = next_word.pos_ if hasattr(next_word, 'pos_') else 'NONE'
        
        if len(next_word_text) >= 3:
            next_last_three = next_word_text[-3:]
        else:
            next_last_three = next_word_text
            
        try:
            next_starts_with_capital = next_word_text[0].isupper() if next_word_text else False
        except:
            next_starts_with_capital = False
        
        features.extend([
            f'word{1}.lower=' + next_word_text.lower(),
            f'word{1}.postag=' + next_word_pos,
            f'word{1}[-3:]=' + next_last_three,
            f'word{1}.isdigit={next_word_text.isdigit()}',
            f'word{1}.startsWithCapital={next_starts_with_capital}'
        ])
    
    if cur_loc == end_loc:
        features.append('END')

    return features

def train_crf_model(X_train, y_train):
    """
    Trains a Conditional Random Field (CRF) model for named entity recognition.
    
    Args:
        X_train: Training features
        y_train: Training labels in BIO format
    
    Returns:
        Trained CRF model
    """
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    crf.fit(X_train, y_train)
    
    return crf

def convert_predictions_to_entities(texts, y_pred):
    """
    Converts BIO tag predictions into entity dictionaries.
    
    Args:
        texts: List of original text strings
        y_pred: Predicted BIO tags from the CRF model
    
    Returns:
        List of dictionaries with extracted entities grouped by entity type
    """
    result = []
    
    for i, text in enumerate(texts):
        doc = nlp(text)
        labels = y_pred[i]
        
        entity_dict = {
            'Condition': [],
            'Procedure': [],
            'Medication': []
        }
        
        current_entity = None
        current_type = None
        current_start = None
        
        for j, (token, label) in enumerate(zip(doc, labels)):
            if label.startswith('B-'):
                if current_entity is not None:
                    entity_dict[current_type].append(current_entity)
                
                current_type = label[2:]
                current_entity = token.text
                current_start = j
            
            elif label.startswith('I-'):
                if current_entity is not None and label[2:] == current_type:
                    current_entity += ' ' + token.text
                else:
                    current_type = label[2:]
                    current_entity = token.text
                    current_start = j
            
            elif label == 'O' and current_entity is not None:
                entity_dict[current_type].append(current_entity)
                current_entity = None
                current_type = None
                current_start = None
        
        if current_entity is not None:
            entity_dict[current_type].append(current_entity)
        
        result.append(entity_dict)
    
    return result

def generate_output_adjusted(texts, predicted_entity_dicts, original_df=None):
    """
    Generates a DataFrame with predicted entities and optionally original annotations.
    
    Args:
        texts: List of original text strings
        predicted_entity_dicts: List of dictionaries with predicted entities
        original_df: Optional DataFrame containing original entity annotations
    
    Returns:
        DataFrame with columns for text, predicted entities, and original entities if provided
    """
    import pandas as pd
    
    results = []
    
    for i, text in enumerate(texts):
        try:
            condition_str = ', '.join(predicted_entity_dicts[i]['Condition']) if predicted_entity_dicts[i]['Condition'] else ''
            procedure_str = ', '.join(predicted_entity_dicts[i]['Procedure']) if predicted_entity_dicts[i]['Procedure'] else ''
            medication_str = ', '.join(predicted_entity_dicts[i]['Medication']) if predicted_entity_dicts[i]['Medication'] else ''
            
            result_row = {
                'text': text,
                'Condition': condition_str,
                'Procedure': procedure_str,
                'Medication': medication_str
            }
            
            if original_df is not None and i < len(original_df):
                result_row['original_Condition'] = ', '.join(original_df.iloc[i]['Condition']) if isinstance(original_df.iloc[i]['Condition'], list) else original_df.iloc[i]['Condition']
                result_row['original_Procedure'] = ', '.join(original_df.iloc[i]['Procedure']) if isinstance(original_df.iloc[i]['Procedure'], list) else original_df.iloc[i]['Procedure']
                result_row['original_Medication'] = ', '.join(original_df.iloc[i]['Medication']) if isinstance(original_df.iloc[i]['Medication'], list) else original_df.iloc[i]['Medication']
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing text #{i}: {str(e)[:100]}...")
            result_row = {
                'text': text,
                'Condition': '',
                'Procedure': '',
                'Medication': ''
            }
            
            if original_df is not None and i < len(original_df):
                result_row['original_Condition'] = ', '.join(original_df.iloc[i]['Condition']) if isinstance(original_df.iloc[i]['Condition'], list) else original_df.iloc[i]['Condition']
                result_row['original_Procedure'] = ', '.join(original_df.iloc[i]['Procedure']) if isinstance(original_df.iloc[i]['Procedure'], list) else original_df.iloc[i]['Procedure']
                result_row['original_Medication'] = ', '.join(original_df.iloc[i]['Medication']) if isinstance(original_df.iloc[i]['Medication'], list) else original_df.iloc[i]['Medication']
            
            results.append(result_row)
    
    return pd.DataFrame(results)

def run_crf_pipeline_with_original_columns(X_train, y_train_entity_dicts, X_test, y_test_entity_dicts, y_test_df=None):
    """
    Runs the complete CRF pipeline: training, evaluation, and prediction.
    
    Args:
        X_train: List of training texts
        y_train_entity_dicts: Training entity annotations
        X_test: List of test texts
        y_test_entity_dicts: Test entity annotations
        y_test_df: Optional DataFrame with original test annotations
    
    Returns:
        Dictionary containing the model, evaluation metrics, predictions, and output DataFrame
    """
    print("Preparing training data for CRF...")
    X_train_features, y_train_labels = prepare_data_BIO(X_train, y_train_entity_dicts, get_features_for_sentence)
    
    print("Training CRF model...")
    crf_model = train_crf_model(X_train_features, y_train_labels)
    
    print("Preparing test data for CRF...")
    X_test_features, y_test_labels = prepare_data_BIO(X_test, y_test_entity_dicts, get_features_for_sentence)
    
    print("Evaluating CRF model...")
    eval_results, report, y_pred = evaluate_crf_model(crf_model, X_test_features, y_test_labels)
    
    print("Converting predictions to entity format...")
    pred_entity_dicts = convert_predictions_to_entities(X_test, y_pred)
    
    print("Generating output dataframe with original columns...")
    output_df = generate_output_adjusted(X_test, pred_entity_dicts, original_df=y_test_df)
    
    return {
        'model': crf_model,
        'evaluation': eval_results,
        'report': report,
        'predictions': pred_entity_dicts,
        'output_df': output_df
    }

def crf_main():
    """
    Main function to run the CRF model pipeline.
    
    This function:
    1. Loads train and test data
    2. Prepares entity dictionaries from the data
    3. Runs the CRF pipeline
    4. Saves the output to a CSV file
    5. Prints detailed performance metrics
    """
    X_train, y_train, X_test, y_test = read_train_test_split()
    train_entities = []
    for i in range(len(y_train)):
        train_entities.append({
            "Condition": y_train.iloc[i]['Condition'],
            "Procedure": y_train.iloc[i]['Procedure'],
            "Medication": y_train.iloc[i]['Medication']
        })

    test_entities = []
    for i in range(len(y_test)):
        test_entities.append({
            "Condition": y_test.iloc[i]['Condition'],
            "Procedure": y_test.iloc[i]['Procedure'],
            "Medication": y_test.iloc[i]['Medication']
        })

    test_df = pd.DataFrame({
        'Condition': y_test['Condition'],
        'Procedure': y_test['Procedure'],
        'Medication': y_test['Medication']
    })

    results = run_crf_pipeline_with_original_columns(
        X_train.tolist(), 
        train_entities, 
        X_test.tolist(), 
        test_entities,
        y_test_df=test_df
    )

    results['output_df'].to_csv(CRF_MODEL_OUTPUT_FILE, index=False)

    print("\nDetailed Classification Report:")
    print(results['report'])

    for entity_type in ['Condition', 'Procedure', 'Medication']:
        if entity_type in results['evaluation']:
            print(f"\n{entity_type}:")
            print(f"F1 Score: {results['evaluation'][entity_type]['f1']:.4f}")
            print(f"Precision: {results['evaluation'][entity_type]['precision']:.4f}")
            print(f"Recall: {results['evaluation'][entity_type]['recall']:.4f}")

    print("\nCRF Model Performance:")
    print(f"Overall F1 Score: {results['evaluation']['overall']['f1']:.4f}")
    print(f"Overall Precision: {results['evaluation']['overall']['precision']:.4f}")
    print(f"Overall Recall: {results['evaluation']['overall']['recall']:.4f}")

# %%
## Uncomment to run the model
# crf_main()

# %%
