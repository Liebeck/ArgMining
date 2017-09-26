#!/usr/bin/env python
# -*- coding: UTF-8 -*-

subtasks = ['A', 'B', 'C']
features = ['unigram', 'bigram', 'character_ngrams', 'pos_distribution_spacy', 'dependency_distribution_spacy',
            'unigram+grammatical_spacy', 'n_unigram+shape', 'sentiws_distribution']
word_embeeding_features = ['embedding_centroid_100', 'embedding_centroid_200', 'embedding_centroid_300']
classifiers = ['svm', 'svm-linear', 'knn', 'rf']

word_embeddings = {'100': 'embeddings_path=/scratch_gs/malie102/data/wikipedia-de/word2vec_wiki-de_20170501_100_binary',
                   '200': 'embeddings_path=/scratch_gs/malie102/data/wikipedia-de/word2vec_wiki-de_20170501_200_binary',
                   '300': 'embeddings_path=/scratch_gs/malie102/data/wikipedia-de/word2vec_wiki-de_20170501_300_binary'}

lda_wikipedia = {
    '100': 'lda_path=/scratch_gs/malie102/data/lda/wikipedia/lda_100.lda,lda_vocab_path=/scratch_gs/malie102/data/lda/wikipedia/_wordids.txt.bz2',
    '200': 'lda_path=/scratch_gs/malie102/data/lda/wikipedia/lda_200.lda,lda_vocab_path=/scratch_gs/malie102/data/lda/wikipedia/_wordids.txt.bz2',
    '300': 'lda_path=/scratch_gs/malie102/data/lda/wikipedia/lda_300.lda,lda_vocab_path=/scratch_gs/malie102/data/lda/wikipedia/_wordids.txt.bz2'}

lda_thf = {
    '5': 'lda_path=/scratch_gs/malie102/data/lda/thf/lda_5,lda_vocab_path=/scratch_gs/malie102/data/lda/thf/lda_5_wordids.txt.bz2',
    '10': 'lda_path=/scratch_gs/malie102/data/lda/thf/lda_10,lda_vocab_path=/scratch_gs/malie102/data/lda/thf/lda_10_wordids.txt.bz2',
    '15': 'lda_path=/scratch_gs/malie102/data/lda/thf/lda_15,lda_vocab_path=/scratch_gs/malie102/data/lda/thf/lda_15_wordids.txt.bz2',
    '20': 'lda_path=/scratch_gs/malie102/data/lda/thf/lda_20,lda_vocab_path=/scratch_gs/malie102/data/lda/thf/lda_20_wordids.txt.bz2',
    '25': 'lda_path=/scratch_gs/malie102/data/lda/thf/lda_25,lda_vocab_path=/scratch_gs/malie102/data/lda/thf/lda_25_wordids.txt.bz2'}

fasttext = {'5': {'3_3': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-5',
                  '4_4': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-5',
                  '5_5': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-5',
                  '6_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-5',
                  '3_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-5'},
            '10': {'3_3': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-10',
                   '4_4': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-10',
                   '5_5': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-10',
                   '6_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-10',
                   '3_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-10'},
            '20': {'3_3': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-20',
                   '4_4': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-20',
                   '5_5': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-20',
                   '6_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-20',
                   '3_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-20'},
            '50': {'3_3': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-50',
                   '4_4': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-50',
                   '5_5': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-50',
                   '6_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-50',
                   '3_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-50'},
            '100': {'3_3': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-100',
                    '4_4': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-100',
                    '5_5': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-100',
                    '6_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-100',
                    '3_6': 'fasttext_path=/scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-100'}
            }


def basic_features(handler, subtask, classifier):
    for feature in features:
        qsub_command = 'qsub -v c={},subtask={},gridsearchstrategy={} hilbert_data_v3.job'.format(classifier, subtask,
                                                                                                  feature)
        handler.write('{}\n'.format(qsub_command))


def get_word_embedding_features(handler, subtask, classifier):
    for feature in word_embeeding_features:
        dimension = feature[-3:]
        qsub_command = 'qsub -v c={},subtask={},gridsearchstrategy={},{} hilbert_data_v3.job'.format(classifier,
                                                                                                     subtask,
                                                                                                     feature,
                                                                                                     word_embeddings[
                                                                                                         dimension])
        handler.write('{}\n'.format(qsub_command))


def get_character_embedding_features(handler, subtask, classifier):
    iterations = [5, 10, 20, 50, 100]
    for iteration in iterations:
        for key, fasttextpath in fasttext[str(iteration)].items():
            qsub_command = 'qsub -v c={},subtask={},gridsearchstrategy=character_embeddings_centroid_100,{} hilbert_data_v3.job'.format(
                classifier,
                subtask,
                fasttextpath)
            handler.write('{}\n'.format(qsub_command))


def get_lda_features(handler, subtask, classifier):
    features_lda = ['lda_distribution', 'unigram+lda_distribution']
    for feature in features_lda:
        for key, lda_path in lda_wikipedia.items():
            qsub_command = 'qsub -v c={},subtask={},gridsearchstrategy={},{} hilbert_data_v3.job'.format(classifier,
                                                                                                         subtask,
                                                                                                         feature,
                                                                                                         lda_path)
            handler.write('{}\n'.format(qsub_command))
        for key, lda_path in lda_thf.items():
            qsub_command = 'qsub -v c={},subtask={},gridsearchstrategy={},{} hilbert_data_v3.job'.format(classifier,
                                                                                                         subtask,
                                                                                                         feature,
                                                                                                         lda_path)
            handler.write('{}\n'.format(qsub_command))


if __name__ == '__main__':
    with open('hilbert_job_list.txt', 'w') as handler:
        for subtask in subtasks:
            for classifier in classifiers:
                basic_features(handler, subtask, classifier)
                get_word_embedding_features(handler, subtask, classifier)
                get_character_embedding_features(handler, subtask, classifier)
                get_lda_features(handler, subtask, classifier)
