import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import classification_report
import spacy
import re
from tqdm import tqdm
from helper import read_train_test_split, prepare_data_BIO

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to extract features for each token in a sentence
def get_features_for_sentence(doc):
    """Extract features for each token in the SpaCy doc."""
    features = []
    for token in doc:
        token_features = {
            'text': token.text,
            'pos': token.pos_,
            'tag': token.tag_,
            'dep': token.dep_,
            'is_alpha': token.is_alpha,
            'is_stop': token.is_stop,
            'is_digit': token.is_digit,
            'is_punct': token.is_punct,
            'shape': token.shape_,
            'prefix': token.prefix_,
            'suffix': token.suffix_,
            'like_num': token.like_num,
            'is_oov': token.is_oov
        }
        features.append(token_features)
    return features


# Custom dataset class for BERT
class MedicalNERDataset(Dataset):
    def __init__(self, texts, entity_dicts, tokenizer, max_len=128):
        self.texts = texts
        self.entity_dicts = entity_dicts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Process data using the prepare_data_BIO function
        _, self.bio_labels = prepare_data_BIO(texts, entity_dicts)
        
        # Create label map
        self.label_map = {
            'O': 0,
            'B-Condition': 1, 'I-Condition': 2,
            'B-Procedure': 3, 'I-Procedure': 4,
            'B-Medication': 5, 'I-Medication': 6
        }
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        bio_labels = self.bio_labels[idx]
        
        # Process text with spaCy to get tokens that match the BIO labels
        doc = nlp(text)
        tokens = [token.text for token in doc]
        
        # Encode text for BERT
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )
        
        # Get token to word mapping
        offset_mapping = encoding['offset_mapping'][0].numpy()
        special_tokens_mask = encoding['special_tokens_mask'][0].numpy()
        
        # Remove offset_mapping and special_tokens_mask from encoding (not needed for model)
        del encoding['offset_mapping']
        del encoding['special_tokens_mask']
        
        # Create aligned labels for BERT tokens
        aligned_labels = []
        current_word_idx = -1
        
        for i, (offset, is_special) in enumerate(zip(offset_mapping, special_tokens_mask)):
            if is_special:
                # Special token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)  # -100 is ignored in loss calculation
            elif offset[0] == 0 and offset[1] != 0:
                # Start of a new word
                current_word_idx += 1
                if current_word_idx < len(bio_labels):
                    label = bio_labels[current_word_idx]
                    aligned_labels.append(self.label_map[label])
                else:
                    # We've gone beyond the tokens we have labels for
                    aligned_labels.append(-100)
            else:
                # Continuation of a word
                if current_word_idx < len(bio_labels):
                    # For continuation tokens, keep I-tag if the token is entity, else ignore
                    label = bio_labels[current_word_idx]
                    if label.startswith('B-'):
                        # Convert B- to I- for continuation tokens
                        i_label = 'I-' + label[2:]
                        aligned_labels.append(self.label_map[i_label])
                    elif label.startswith('I-'):
                        # Keep I- tag
                        aligned_labels.append(self.label_map[label])
                    else:
                        # Ignore continuation of 'O' tokens
                        aligned_labels.append(-100)
                else:
                    aligned_labels.append(-100)
        
        # Make sure we have the right length
        if len(aligned_labels) < self.max_len:
            aligned_labels.extend([-100] * (self.max_len - len(aligned_labels)))
        elif len(aligned_labels) > self.max_len:
            aligned_labels = aligned_labels[:self.max_len]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels)
        }

def train_model(model, train_dataloader, val_dataloader, device, epochs=1):
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Get the id2label mapping
    id2label = model.config.id2label
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        # Training loop
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        
        # Validation loop
        model.eval()
        val_loss = 0
        predictions, true_labels = [], []
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                pred_ids = torch.argmax(logits, dim=2)
                
                # Convert predictions and labels to tag sequences
                for i in range(input_ids.shape[0]):
                    pred_seq = []
                    true_seq = []
                    
                    for j in range(input_ids.shape[1]):
                        if labels[i, j] != -100:
                            pred_label = id2label[pred_ids[i, j].item()]
                            true_label = id2label[labels[i, j].item()]
                            
                            pred_seq.append(pred_label)
                            true_seq.append(true_label)
                    
                    predictions.append(pred_seq)
                    true_labels.append(true_seq)
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss}")
        
        # Calculate metrics
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
    
    return model

def predict_entities(text, model, tokenizer, device):
    """Predict entities in a given text using the trained model."""
    # Process text with spaCy
    doc = nlp(text)
    tokens = [token.text for token in doc]
    
    # Encode text for BERT
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    
    # Get offset mapping and remove it from inputs
    offset_mapping = encoding.pop('offset_mapping')[0].cpu().numpy()
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in encoding.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    # Convert predictions to entity spans
    entities = []
    current_entity = None
    id2label = model.config.id2label
    
    for i, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens
        if offset[0] == 0 and offset[1] == 0:
            continue
            
        # Get the predicted label
        label = id2label[pred_id]
        
        # Start of a new entity
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            entity_type = label[2:]  # Extract entity type
            start = offset[0].item()
            end = offset[1].item()
            current_entity = {"type": entity_type, "start": start, "end": end}
        
        # Inside an entity
        elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
            current_entity["end"] = offset[1].item()
            
        # Outside any entity
        elif label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add the last entity if there is one
    if current_entity:
        entities.append(current_entity)
    
    # Extract the entity text and create result
    result = []
    for entity in entities:
        entity_text = text[entity["start"]:entity["end"]]
        result.append({
            "text": entity_text,
            "type": entity["type"],
            "start": entity["start"],
            "end": entity["end"]
        })
    
    return result

def per_entity_eval(test_true_labels, test_predictions):
    report = classification_report(test_true_labels, test_predictions, output_dict=True)
    print("\nPer-entity evaluation:")
    for entity_type in ["Condition", "Procedure", "Medication"]:
        b_entity = f"B-{entity_type}"
        i_entity = f"I-{entity_type}"
        
        # Check if the entity exists in the report (might not if not predicted)
        if b_entity in report:
            b_precision = report[b_entity]['precision']
            b_recall = report[b_entity]['recall']
            b_f1 = report[b_entity]['f1-score']
            b_support = report[b_entity]['support']
            
            print(f"\n{entity_type}:")
            print(f"  Precision: {b_precision:.3f}")
            print(f"  Recall: {b_recall:.3f}")
            print(f"  F1 Score: {b_f1:.3f}")
            print(f"  Support: {b_support}")
            
            # Also check for I- tags
            if i_entity in report:
                i_precision = report[i_entity]['precision']
                i_recall = report[i_entity]['recall']
                i_f1 = report[i_entity]['f1-score']
                i_support = report[i_entity]['support']
                
                print(f"  I-tag metrics:")
                print(f"    Precision: {i_precision:.3f}")
                print(f"    Recall: {i_recall:.3f}")
                print(f"    F1 Score: {i_f1:.3f}")
                print(f"    Support: {i_support}")
        else:
            print(f"\n{entity_type}: No predictions found")
    
def bert_main():
    # Load your data using your existing function
    X_train, y_train, X_test, y_test = read_train_test_split()
    
    # Convert dataframes to list of dictionaries
    train_entities = y_train.to_dict('records')
    train_texts = X_train.tolist()
    
    # Use a portion of test data as validation
    val_texts, test_texts, val_entities, test_entities = train_test_split(
        X_test.tolist(), y_test.to_dict('records'), test_size=0.5, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Create label map
    label_map = {
        'O': 0,
        'B-Condition': 1, 'I-Condition': 2,
        'B-Procedure': 3, 'I-Procedure': 4,
        'B-Medication': 5, 'I-Medication': 6
    }
    id2label = {v: k for k, v in label_map.items()}
    
    # Create datasets using the prepare_data_BIO approach
    train_dataset = MedicalNERDataset(train_texts, train_entities, tokenizer)
    val_dataset = MedicalNERDataset(val_texts, val_entities, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = BertForTokenClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_map),
        id2label=id2label,
        label2id=label_map
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Train the model
    model = train_model(model, train_dataloader, val_dataloader, device, epochs=3)
    
    # Save the model
    model.save_pretrained('./medical_ner_model')
    tokenizer.save_pretrained('./medical_ner_model')
    
    # Test the model on the test set
    print("\nEvaluating on test set:")
    test_dataset = MedicalNERDataset(test_texts, test_entities, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    
    # Evaluation loop
    model.eval()
    test_predictions, test_true_labels = [], []
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get predictions
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=2)
            
            # Convert predictions and labels to tag sequences
            for i in range(input_ids.shape[0]):
                pred_seq = []
                true_seq = []
                
                for j in range(input_ids.shape[1]):
                    if labels[i, j] != -100:
                        pred_label = id2label[pred_ids[i, j].item()]
                        true_label = id2label[labels[i, j].item()]
                        
                        pred_seq.append(pred_label)
                        true_seq.append(true_label)
                
                test_predictions.append(pred_seq)
                test_true_labels.append(true_seq)
    
    # Calculate metrics
    test_precision = precision_score(test_true_labels, test_predictions)
    test_recall = recall_score(test_true_labels, test_predictions)
    test_f1 = f1_score(test_true_labels, test_predictions)
    
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")
    print(f"Test F1 Score: {test_f1:.3f}")
    
    per_entity_eval(test_true_labels, test_predictions)
bert_main()
