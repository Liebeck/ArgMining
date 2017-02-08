# ArgMining

## Sentence-level scripts
### cross_validation.py
Performs a cross-validation for a strategy with fixed feature parameters on the training set
``` bash
  python scripts/sentence/cross_validation.py -subtask A -strategy unigram -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy pos_distribution -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy sentiws_polarity -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy embedding_centroid_100 -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy embedding_centroid_stopwords_100 -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy n_unigram+pos_distribution+embedding_centroid -c svm


```

### gridsearch.py
Performs a cross-validation on the training set with a feature combination and experiments with different parameters for the features. The result of the gridsearch is saved in the file system.
``` bash
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy unigram -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy bigram -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy pos_distribution_feature_selection -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy pos_distribution -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy unigram+grammatical -c svm
    python scripts/sentence/gridsearch.py -subtask B -gridsearchstrategy unigram+grammatical -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy embedding_centroid_100 -c svm


```

### evaluate.py
Given a path to a settings file, the evaluate script trains the specified classiers on the training set and predicts on the test set.
``` bash
    python scripts/sentence/evaluate.py -configfile results/sentence/temp/XXX

```