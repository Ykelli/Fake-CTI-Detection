## Updates
- 06-Nov-2023: released


## 1. Set-up
### environment requirements:
python = 3.7
```
pip install -r requirements.txt
```

## 2. Generate features from xlsx
### prepare resources
```
cd ./model
python -m spacy download en_core_web_lg
```

### Download embedding models file 
Download GoogleNews-vectors-negative300.bin.gz file into ./model from https://github.com/mmihaltz/word2vec-GoogleNews-vectors
Download word2vec_million_cybersecurity_docs_models.tar.gz into ./model from link on https://github.com/UMBC-ACCL/cybersecurity_embeddings, and unzip to ./model. (will get 3 files including 1million.word2vec.model in ./model)
Download Domain-Word2vec.model.gz into ./model from link on https://github.com/Ebiquity/CASIE/tree/master/embeddings

### generate features
```
cd ..
python ./genarate_features_from_dataset.py --input dataset_long.xlsx --output dataset_long_with_feature.xlsx
```

## 3. Feature-based analyse
### feature analyse
```
python ./analyse_on_features.py --input dataset_long_with_feature.xlsx
```

## 4. Text-based analyse

### SVM
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function passive_aggressive
```

### Passive Aggressive Classifier
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function passive_aggressive
```

### Logistic Regression Classifier
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function logic_regression
```

### Random forest Classifier
```
python ./analyse_on_text.py --input dataset_long_with_feature.xlsx --function random_forest
```