# ArgMining
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Liebeck/ArgMining/blob/master/LICENSE.md)  
This machine learning project contains the latest code version for the Argument Mining tasks researched by Matthias Liebeck as part of his PhD thesis.

## Requirements
* [docker](https://www.docker.com/)
* Execute the *data/download_resources.sh* script to download the models
``` bash
./data/download_resources.sh data

```
* Depending on whether you want to use our trained Wikipedia word2vec models, you also need to download word embeddings via this script:
``` bash
./data/download_embeddings.sh data

```