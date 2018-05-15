# ArgMining
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Liebeck/ArgMining/blob/master/LICENSE.md)  
This machine learning project contains the latest code version for the Argument Mining tasks researched by Matthias Liebeck as part of his PhD thesis.

``` bash
The code of this project is archived to reflect the state of the PhD submission. Further development and refactoring can be found [here](https://github.com/Liebeck/textclassification).
```


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

* If you want to use character embeddings from fasttext, you can download the specific model here: http://lager.cs.uni-duesseldorf.de/NLP/fasttext/german/wikipedia/de-wiki_20170501/  
With the following script, you can download chracter (3, 3)-grams that were trained with 5 and with 50 iterations:

``` bash
./data/download_character_embeddings.sh data
```