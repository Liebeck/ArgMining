#!/bin/bash

if [ -z "$1" ]
  then
    echo "No download directory specified"
fi

mkdir -p $1/fasttext


function download_character_embeddings {
    # parameter 1: directory
    # parameter 2: download file
    echo $1
    echo $2
    file="${2##*/}"
    wget -O $1/${file} $2
    wget -O $1/${file}.wv.syn0.npy $2.wv.syn0.npy
    wget -O $1/${file}.wv.syn0_all.npy $2.wv.syn0_all.npy
}


download_character_embeddings "$1/fasttext" "http://lager.cs.uni-duesseldorf.de/NLP/fasttext/german/wikipedia/de-wiki_20170501/dewiki-20170501-3_3-5"
download_character_embeddings "$1/fasttext" "http://lager.cs.uni-duesseldorf.de/NLP/fasttext/german/wikipedia/de-wiki_20170501/dewiki-20170501-3_3-50"



