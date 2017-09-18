
## Sentence-level scripts
### cross_validation.py
Performs a cross-validation for a strategy with fixed feature parameters on the training set
``` bash
  python scripts/sentence/cross_validation.py -subtask A -strategy unigram -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy pos_distribution_spacy -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy sentiws_polarity -c svm
  python scripts/sentence/cross_validation.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -strategy embedding_centroid_100 -c svm
  python scripts/sentence/cross_validation.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -strategy embedding_centroid_stopwords_100 -c svm
  python scripts/sentence/cross_validation.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -strategy n_unigram+pos_distribution+embedding_centroid -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy unigram_lowercase_tfidf -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy n_unigram_shape -c svm
  python scripts/sentence/cross_validation.py -subtask A -strategy n_unigram_shape_lemma -c svm


```

### gridsearch.py
Performs a cross-validation on the training set with a feature combination and experiments with different parameters for the features. The result of the gridsearch is saved in the file system.
``` bash
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy unigram -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy bigram -c svm
    # python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy pos_distribution_feature_selection -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy pos_distribution_spacy -c svm
    python scripts/sentence/gridsearch.py -subtask A -gridsearchstrategy unigram+grammatical_spacy -c svm
    python scripts/sentence/gridsearch.py -subtask B -gridsearchstrategy unigram+grammatical_spacy -c svm
    python scripts/sentence/gridsearch.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -gridsearchstrategy embedding_centroid_100 -c svm
    python scripts/sentence/gridsearch.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -nfold 10 -gridsearchstrategy embedding_centroid_100 -c svm
    python scripts/sentence/gridsearch.py -subtask A -embeddings_path data/word_embeddings/word2vec_wiki-de_20161120_100_binary -nfold 10 -gridsearchstrategy unigram+embedding_centroid_100 -c svm

```

### evaluate.py
Given a path to a settings file, the evaluate script trains the specified classiers on the training set and predicts on the test set.
``` bash
    python scripts/sentence/evaluate.py -configfile results/sentence/temp/XXX

```


### Hilbert
Data_v1
``` bash
qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v1.job
qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v1.job
qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v1.job
qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v1.job
qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v1.job
qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v1.job

qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v1.job
qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v1.job
qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v1.job

```

Data_v2
``` bash
qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v2.job
qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v2.job
qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v2.job
qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v2.job
qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v2.job
qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v2.job

qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v2.job
qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v2.job
qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v2.job

```