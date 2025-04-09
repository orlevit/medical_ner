from rule_based_model import rule_base_main
from train_test_split import train_test_splitting
from strubell_model import spacy_strubell_model_main
from bert_model import bert_main

print(f'------------------------- Train-Test Split --------------------------')
train_test_splitting()
print(f'------------------------- Rule-base model --------------------------')
rule_base_main()
print(f'------------------------ Strubell model --------------------------')
spacy_strubell_model_main()
print(f'------------------------ bert model --------------------------')
bert_main()
