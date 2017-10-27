# dev.py

Local
``` bash
python3 scripts/deep_learning/dev.py -subtask A -padding_length 20
python3 scripts/deep_learning/dev.py -subtask A -padding_length 20 -embedding_cache_name word2vec_wiki_de_20170501_300-reduced

python3 scripts/deep_learning/run_single_benchmark.py -subtask A -configpath results/sentence_deeplearning/benchmarks/lstm-embedding-empty_001.json

python3 scripts/deep_learning/run_all_benchmarks.py -directory results/sentence_deeplearning/benchmarks/


```

# create_embedding_cache
``` bash
python3 scripts/deep_learning/create_embedding_cache.py -embedding_type word2vec -embedding_path /home/matthias/shared/word2vec/word2vec_wiki-de_20170501_300_binary -embedding_cache_name word2vec_wiki_de_20170501_300-reduced
python3 scripts/deep_learning/create_embedding_cache.py -embedding_type word2vec -embedding_path /home/matthias/shared/word2vec/word2vec_wiki-de_20170501_300_binary -embedding_cache_name word2vec_wiki_de_20170501_300-reduced-both
python3 scripts/deep_learning/create_embedding_cache.py -embedding_type fasttext -embedding_path /home/matthias/shared/fasttext/dewiki-20170501-3_6-10 -embedding_cache_name fasttext_dewiki-20170501-3_6-10-reduced
```






Word2vec cache: 6188/7065 words
fastText cache: 6890/7065 words
