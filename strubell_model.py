import re
import spacy
from spacy.util import minibatch,compounding
from spacy.training import Example
import random
import warnings
import pandas as pd
from config import STRUBELL_MODEL_OUTPUT_FILE
from helper import read_train_test_split
warnings.filterwarnings('ignore')

def convert_to_spacy_format(texts, entities_lists):
    training_data = []
    
    valid_entity_types = {"Condition", "Procedure", "Medication"}
    
    for i, text in enumerate(texts):
        candidate_spans = []
        for entity_type, entity_list in entities_lists[i].items():
            if entity_type not in valid_entity_types:
                continue
                
            for entity in entity_list:
                if not entity:
                    continue
                    
                entity_lower = entity.lower()
                text_lower = text.lower()
                
                for match in re.finditer(re.escape(entity_lower), text_lower):
                    start, end = match.span()
                    candidate_spans.append((start, end, entity_type, end - start))
        
        candidate_spans.sort(key=lambda x: x[3], reverse=True)
        
        final_spans = []
        token_occupancy = set()
        
        for start, end, entity_type, _ in candidate_spans:
            span_positions = set(range(start, end))
            if span_positions.intersection(token_occupancy):
                continue
            
            final_spans.append((start, end, entity_type))
            token_occupancy.update(span_positions)
        
        final_spans.sort(key=lambda x: x[0])
        
        training_data.append((text, {"entities": final_spans}))
    
    return training_data

def train_spacy_model(model, train_data, iterations=30):
    losses = {}
    
    if "ner" not in model.pipe_names:
        ner = model.add_pipe("ner")
    else:
        ner = model.get_pipe("ner")
        
    added_labels = set()
    for text, annotations in train_data:
        for ent in annotations.get("entities", []):
            if len(ent) == 3:
                label = ent[2]
                if label not in added_labels:
                    ner.add_label(label)
                    added_labels.add(label)
    
    
    pipe_exceptions = ["ner", "tok2vec"]
    unaffected_pipes = [pipe for pipe in model.pipe_names if pipe not in pipe_exceptions]

    with model.disable_pipes(*unaffected_pipes):
        examples = []
        skipped_count = 0
        
        for idx, (text, annots) in enumerate(train_data):
            try:
                doc = model.make_doc(text)
                
                if len(doc) == 0:
                    print(f"Skipping example #{idx}: Empty document")
                    skipped_count += 1
                    continue
                
                invalid_spans = []
                for start, end, label in annots.get("entities", []):
                    if start >= end or start < 0 or end > len(text):
                        invalid_spans.append((start, end, label))
                
                if invalid_spans:
                    print(f"Skipping example #{idx}: Contains invalid spans {invalid_spans}")
                    skipped_count += 1
                    continue
                
                example = Example.from_dict(doc, annots)
                examples.append(example)
                
            except ValueError as e:
                print(f"Skipping example #{idx}: {str(e)[:100]}...")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"Unexpected error in example #{idx}: {str(e)[:100]}...")
                skipped_count += 1
                continue
        
        
        if not examples:
            raise ValueError("No valid training examples were created. Cannot train model.")
        
        optimizer = model.begin_training()
        for i in range(iterations):
            random.shuffle(examples)
            
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            
            batch_losses = {}
            for batch in batches:
                try:
                    model.update(
                        batch,
                        drop=0.5,
                        sgd=optimizer,
                        losses=batch_losses
                    )
                    for k, v in batch_losses.items():
                        if k not in losses:
                            losses[k] = [v]
                        else:
                            losses[k].append(v)
                except Exception as e:
                    print(f"Error during batch update: {str(e)[:100]}...")
                    continue
            
            iter_loss = {}
            for k, v in losses.items():
                if v:
                    iter_loss[k] = sum(v) / len(v)
            
            print(f"Iteration {i+1}/{iterations}, Loss: {iter_loss}")
    
    return model


def evaluate_model(model, test_data):
    """
    Evaluate the model on test data and calculate precision, recall, and F1 score.
    
    Args:
        model: Trained spaCy model
        test_data: Test data in spaCy format
    
    Returns:
        Dictionary with evaluation metrics
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for text, annotations in test_data:
        true_entities = annotations.get("entities", [])
        true_entity_spans = set((start, end, label) for start, end, label in true_entities)
        
        doc = model(text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        pred_entity_spans = set(pred_entities)
        
        # True positives: entities that are both in true and predicted sets
        true_positives += len(true_entity_spans.intersection(pred_entity_spans))
        
        # False positives: entities in predicted but not in true set
        false_positives += len(pred_entity_spans - true_entity_spans)
        
        # False negatives: entities in true but not in predicted set
        false_negatives += len(true_entity_spans - pred_entity_spans)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def generate_output(model, texts, original_df=None):
    
    results = []
    
    for i, text in enumerate(texts):
        try:
            doc = model(text)
            
            conditions = []
            procedures = []
            medications = []
            
            for ent in doc.ents:
                normalized_text = ent.text.strip()
                
                if ent.label_ == "Condition" and normalized_text:
                    conditions.append(normalized_text)
                elif ent.label_ == "Procedure" and normalized_text:
                    procedures.append(normalized_text)
                elif ent.label_ == "Medication" and normalized_text:
                    medications.append(normalized_text)
            
            conditions = list(dict.fromkeys(conditions))
            procedures = list(dict.fromkeys(procedures))
            medications = list(dict.fromkeys(medications))
            
            condition_str = ', '.join(conditions) if conditions else ''
            procedure_str = ', '.join(procedures) if procedures else ''
            medication_str = ', '.join(medications) if medications else ''
            
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

def evaluate_by_entity_type(model, test_data):
    """
    Evaluate the model on test data and calculate metrics for each entity type.
    
    Args:
        model: Trained spaCy model
        test_data: Test data in spaCy format
    
    Returns:
        Dictionary with evaluation metrics by entity type
    """
    metrics_by_type = {
        "Condition": {"tp": 0, "fp": 0, "fn": 0},
        "Procedure": {"tp": 0, "fp": 0, "fn": 0},
        "Medication": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    for text, annotations in test_data:
        true_entities = annotations.get("entities", [])
        doc = model(text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        
        # Group true entities by type
        true_by_type = {
            "Condition": set(),
            "Procedure": set(),
            "Medication": set()
        }
        
        for start, end, label in true_entities:
            true_by_type[label].add((start, end))
        
        # Group predicted entities by type
        pred_by_type = {
            "Condition": set(),
            "Procedure": set(),
            "Medication": set()
        }
        
        for start, end, label in pred_entities:
            if label in pred_by_type:
                pred_by_type[label].add((start, end))
        
        # Calculate metrics for each type
        for entity_type in ["Condition", "Procedure", "Medication"]:
            true_spans = true_by_type[entity_type]
            pred_spans = pred_by_type[entity_type]
            
            metrics_by_type[entity_type]["tp"] += len(true_spans.intersection(pred_spans))
            metrics_by_type[entity_type]["fp"] += len(pred_spans - true_spans)
            metrics_by_type[entity_type]["fn"] += len(true_spans - pred_spans)
    
    # Calculate precision, recall, and F1 for each entity type
    results = {}
    for entity_type, counts in metrics_by_type.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return results


def spacy_strubell_model_main():
    print("Starting Strubell model training..")
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

    train_data = convert_to_spacy_format(X_train.tolist(), train_entities)
    test_data = convert_to_spacy_format(X_test.tolist(), test_entities)

    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner", "tok2vec"]

    trained_model = train_spacy_model(nlp, train_data, iterations=1)

    overall_metrics = evaluate_model(trained_model, test_data)
    entity_metrics = evaluate_by_entity_type(trained_model, test_data)

    print("\nOverall Model Performance:")
    print(f"Precision: {overall_metrics['precision']:.4f}")
    print(f"Recall: {overall_metrics['recall']:.4f}")
    print(f"F1 Score: {overall_metrics['f1']:.4f}")

    print("\nPerformance by Entity Type:")
    for entity_type, metrics in entity_metrics.items():
        print(f"\n{entity_type}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

    output_df = generate_output(trained_model, X_test.tolist(), original_df=y_test)
    output_df.to_csv(STRUBELL_MODEL_OUTPUT_FILE, index=False)
