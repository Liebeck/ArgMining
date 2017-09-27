### Hilbert Resources

Paths to word embeddings representations
``` bash
embeddings_path=/scratch_gs/malie102/data/wikipedia-de/word2vec_wiki-de_20170501_100_binary
```

LDA wikipedia
``` bash
lda_path=/scratch_gs/malie102/data/lda/wikipedia/lda_100.lda
lda_vocab_path=/scratch_gs/malie102/data/lda/wikipedia/_wordids.txt.bz2
```

LDA THF
``` bash
lda_path=/scratch_gs/malie102/data/lda/thf/lda_10
lda_vocab_path=/scratch_gs/malie102/data/lda/thf/lda_10_wordids.txt.bz2
```

fastText character embeddings
``` bash
fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-10
```

### Old Hilbert Scripts

Data_v2
``` bash
#qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v2.job
#qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v2.job
#qsub -v c=svm,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v2.job
#qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v2.job
#qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v2.job
#qsub -v c=knn,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v2.job

#qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram+embedding_centroid_100 hilbert_data_v2.job
#qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=unigram hilbert_data_v2.job
#qsub -v c=rf,subtask=A,embeddings_path=/scratch_gs/malie102/word_embeddings/word2vec_wiki-de_20161120_100_binary,gridsearchstrategy=embedding_centroid_100 hilbert_data_v2.job

```
