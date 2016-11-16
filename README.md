# ArgMining

## Sentence-level scripts
### cross_validation.py
Performs a cross-validation for a strategy with fixed feature parameters on the training set
``` bash
  python scripts/sentence/cross_validation.py -subtask A -strategy unigram -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy pos_distribution -c svm

```

### gridsearch.py
Performs a cross-validation on the training set with a feature combination and experiments with different parameters for the features. The result of the gridsearch is saved in the file system.
``` bash
    python scripts/sentence/gridsearch.py -subtask A -strategy pos_distribution -c svm

```

### evaluate.py
Given a path to a settings file, the evaluate script trains the specified classiers on the training set and predicts on the test set.
``` bash
    python scripts/sentence/predict.py -configfile results/sentence/temp/XXX

```