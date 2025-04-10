# Named Entity Recognition (NER) for Medical Entity Extraction
This project implements three distinct Named Entity Recognition (NER) methods for extracting medical entities (conditions, procedures, and medications) from clinical texts. The approaches include a rule-based system, Conditional Random Fields (CRF), and Iterated Dilated Convolutional Neural Networks (ID-CNNs).

## Table of Contents
1. [Overview](#overview)
2. [High Level Logic](#high-level-logic)
3. [Components](#components)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Tools & Technologies](#tools--technologies)
6. [Results](#results)

## Overview
This project focuses on extracting critical medical entities from clinical texts using different NER techniques. The implemented models achieve high performance, with the best model reaching an F1 score of 0.9540, demonstrating the effectiveness of the approaches for medical text analysis.

Please find the attached PDF for comprehensive report.

---
## High Level Logic:
1. **Exploratory Data Analysis:**  
   Analysis of clinical text characteristics and entity distributions
2. **Data Processing:**  
   Preprocessing of clinical texts, tokenization, and preparation for different model architectures.
3. **Model Implementation:**  
   Development of three distinct NER approaches:
   - Rule-based system using pattern matching and medical dictionaries
   - Conditional Random Fields (CRF) with handcrafted features
   - Iterated Dilated Convolutional Neural Networks (ID-CNNs)
4. **Model Evaluation:**  
   Comprehensive assessment of model performance using precision, recall, and F1 metrics.
5. **Results Analysis:**  
   Comparison of approaches and identification of the most effective techniques.

## Components
- **config.py**: Configuration parameters for models and data processing
- **helper.py**: Utility functions used across the project
- **main.py**: Main execution script for running all models
- **eda.ipynb**: Exploratory data analysis scripts
- **rule_based_model.py**: Implementation of the rule-based approach
- **crf.py** Implementation of the CRF model
- **ID_CNNs_model.py** Implementation of the ID-CNNs model
- **train_test_split.py**  Data splitting into train & test
- **data/output/models_results**: Directory containing the models CSV results.

## Getting Started
### Prerequisites
Python version 3.12.3 

Virtual environment

### Installation
1. **Clone the Repository:**
   ```bash
   git https://github.com/orlevit/medical_ner.git
   cd medical_ner
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute the main script:**
   ```bash
   python main.py
   ```
* The main script aplit the data and runs all the models one by one.
* For each file there is corresponding notebook to view it, if you wish to run only specific model, you can enter the notebook and unmark the last comment.

## Tools & Technologies
- **Python**: Core programming language
- **Scikit-learn**: Used for evaluation metrics and CRF implementation
- **NLTK/spaCy**: Natural language processing tools for text preprocessing
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Matplotlib/Seaborn**: Visualization of results and data distributions

### Results
The results form ech model is provided in "data/output/models results/" directory.
in addition to the predicted entities also the original columns were added for ease of compersion
