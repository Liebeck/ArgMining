# ArgMining

## Test features
``` bash
  python scripts/sentence/cross_validation.py -subtask A -strategy unigram -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy pos_distribution -c svm

```

## Gridsearch
``` bash
    python scripts/sentence/gridsearch.py -subtask A -strategy pos_distribution -c svm

```