from rule_based_model import rule_base_main
from train_test_split import train_test_splitting
from ID_CNNs_model import spacy_strubell_model_main
from crf import crf_main

print(f'------------------------- Train-Test Split --------------------------')
train_test_splitting()
print(f'------------------------- Rule-base model --------------------------')
rule_base_main()
print(f'------------------------- CRF model --------------------------')
crf_main()
print(f'------------------------ ID-CNNs model --------------------------')
spacy_strubell_model_main()
