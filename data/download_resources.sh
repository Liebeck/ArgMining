#!/bin/bash

if [ -z "$1" ]
  then
    echo "No download directory specified"
fi
mkdir -p $1

function download_and_unzip {
    # parameter 1: directory
    # parameter 2: download file
    mkdir -p $1
    wget -O $1/tmp.zip $2
    unzip $1/tmp.zip -d $1
    rm $1/tmp.zip
}


download_and_unzip "$1/IWNLP" "http://lager.cs.uni-duesseldorf.de/NLP/IWNLP/IWNLP.Lemmatizer_20170501.zip"
download_and_unzip "$1/THF" "http://lager.cs.uni-duesseldorf.de/NLP/argmining/THF.ZIP"
download_and_unzip "$1/sentiws" "http://pcai056.informatik.uni-leipzig.de/downloads/etc/SentiWS/SentiWS_v1.8c.zip"

rm $1/sentiws/SentiWS.txt
rm -r $1/sentiws/__MACOSX