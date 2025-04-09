import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR =os.path.join(DATA_DIR, "input") 
OUTPUT_DIR =os.path.join(DATA_DIR, "output") 
MODELS_OUTPUT_DIR =os.path.join(OUTPUT_DIR, "models_results") 

# Files
DATA_FILE = os.path.join(INPUT_DIR, 'NER_dataset.csv') 
TRAIN_TEST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'train_test.json') 
RULE_BASE_MODEL_OUTPUT_FILE = os.path.join(MODELS_OUTPUT_DIR, 'rule_based.csv') 
STRUBELL_MODEL_OUTPUT_FILE = os.path.join(MODELS_OUTPUT_DIR, 'strubell.csv') 
CRF_MODEL_OUTPUT_FILE = os.path.join(MODELS_OUTPUT_DIR, 'crf.csv') 

# Hyperparameters
EMPTY_HIST_THRESHOLD_PROCEDURE = 0.5
EMPTY_HIST_THRESHOLD_CONDITION = 1
EMPTY_HIST_THRESHOLD_MEDICATION = 0.1
