#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from argmining.sentence.loaders.THF_sentence_corpus_loader import load

taskA_train = load(file_path='data/THF/sentence/subtaskA_train.json')
taskA_test = load(file_path='data/THF/sentence/subtaskA_test.json')
taskB_train = load(file_path='data/THF/sentence/subtaskB_train.json')
taskB_test = load(file_path='data/THF/sentence/subtaskB_test.json')
