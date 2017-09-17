#!/bin/bash

if [ -z "$1" ]
  then
    echo "No download directory specified"
fi

mkdir -p $1/word_embeddings


function download_embeddings {
    # parameter 1: directory
    # parameter 2: download file
    echo $1
    echo $2
    file="${2##*/}"
    wget -O $1/$file $2
    wget -O $1/$file_binary $2_binary
    wget -O $1/$file_binary.syn0.npy $2_binary.syn0.npy
    wget -O $1/$file_binary.syn1neg.npy $2_binary.syn1neg.npy
}


download_embeddings "$1/word_embeddings" "http://lager.cs.uni-duesseldorf.de/NLP/word-embeddings/german/de-wiki_20161120/100/word2vec_wiki-de_20161120_100"
download_embeddings "$1/word_embeddings" "http://lager.cs.uni-duesseldorf.de/NLP/word-embeddings/german/de-wiki_20161120/200/word2vec_wiki-de_20161120_200"
download_embeddings "$1/word_embeddings" "http://lager.cs.uni-duesseldorf.de/NLP/word-embeddings/german/de-wiki_20161120/300/word2vec_wiki-de_20161120_300"

