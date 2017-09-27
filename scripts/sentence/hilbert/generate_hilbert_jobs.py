#!/usr/bin/env python
# -*- coding: UTF-8 -*-

subtasks = ['A', 'B', 'C']
features = ['unigram', 'bigram', 'character_ngrams', 'pos_distribution_spacy', 'dependency_distribution_spacy',
            'unigram+grammatical_spacy', 'n_unigram+shape', 'sentiws_distribution']
word_embeeding_features = ['embedding_centroid_100', 'embedding_centroid_200', 'embedding_centroid_300']
classifiers = ['svm', 'svm-linear', 'knn', 'rf']

word_embeddings = {
    '100': '-embeddings_path /scratch_gs/malie102/data/wikipedia-de/word2vec_wiki-de_20170501_100_binary',
    '200': '-embeddings_path /scratch_gs/malie102/data/wikipedia-de/word2vec_wiki-de_20170501_200_binary',
    '300': '-embeddings_path /scratch_gs/malie102/data/wikipedia-de/word2vec_wiki-de_20170501_300_binary'}

lda_wikipedia = {
    '100': '-lda_path /scratch_gs/malie102/data/lda/wikipedia/lda_100.lda -lda_vocab_path /scratch_gs/malie102/data/lda/wikipedia/_wordids.txt.bz2',
    '200': '-lda_path /scratch_gs/malie102/data/lda/wikipedia/lda_200.lda -lda_vocab_path /scratch_gs/malie102/data/lda/wikipedia/_wordids.txt.bz2',
    '300': '-lda_path /scratch_gs/malie102/data/lda/wikipedia/lda_300.lda -lda_vocab_path /scratch_gs/malie102/data/lda/wikipedia/_wordids.txt.bz2'}

lda_thf = {
    '5': '-lda_path /scratch_gs/malie102/data/lda/thf/lda_5 -lda_vocab_path /scratch_gs/malie102/data/lda/thf/lda_5_wordids.txt.bz2',
    '10': '-lda_path /scratch_gs/malie102/data/lda/thf/lda_10 -lda_vocab_path /scratch_gs/malie102/data/lda/thf/lda_10_wordids.txt.bz2',
    '15': '-lda_path /scratch_gs/malie102/data/lda/thf/lda_15 -lda_vocab_path /scratch_gs/malie102/data/lda/thf/lda_15_wordids.txt.bz2',
    '20': '-lda_path /scratch_gs/malie102/data/lda/thf/lda_20 -lda_vocab_path /scratch_gs/malie102/data/lda/thf/lda_20_wordids.txt.bz2',
    '25': '-lda_path /scratch_gs/malie102/data/lda/thf/lda_25 -lda_vocab_path /scratch_gs/malie102/data/lda/thf/lda_25_wordids.txt.bz2'}

fasttext = {'5': {'3_3': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-5',
                  '4_4': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-5',
                  '5_5': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-5',
                  '6_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-5',
                  '3_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-5'},
            '10': {'3_3': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-10',
                   '4_4': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-10',
                   '5_5': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-10',
                   '6_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-10',
                   '3_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-10'},
            '20': {'3_3': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-20',
                   '4_4': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-20',
                   '5_5': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-20',
                   '6_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-20',
                   '3_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-20'},
            '50': {'3_3': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-50',
                   '4_4': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-50',
                   '5_5': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-50',
                   '6_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-50',
                   '3_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-50'},
            '100': {'3_3': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-100',
                    '4_4': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-100',
                    '5_5': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-100',
                    '6_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-100',
                    '3_6': ' -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-100'}
            }


def get_basic_features():
    parameters = []
    for feature in features:
        parameters.append('-gridsearchstrategy {}'.format(feature))
    return parameters


def get_word_embedding_features():
    parameters = []
    for feature in word_embeeding_features:
        dimension = feature[-3:]
        parameters.append('-gridsearchstrategy {}'.format(word_embeddings[dimension]))
    return parameters


def get_character_embedding_features():
    parameters = []
    iterations = [5, 10, 20, 50, 100]
    for iteration in iterations:
        for key, fasttextpath in fasttext[str(iteration)].items():
            parameters.append('-gridsearchstrategy character_embeddings_centroid_100 {}'.format(fasttextpath))
    return parameters


def get_lda_features():
    parameters = []
    features_lda = ['lda_distribution', 'unigram+lda_distribution']
    for feature in features_lda:
        for key, lda_path in lda_wikipedia.items():
            parameters.append('-gridsearchstrategy {} {}'.format(feature, lda_path))
        for key, lda_path in lda_thf.items():
            parameters.append('-gridsearchstrategy {} {}'.format(feature, lda_path))
    return parameters


def create_python_call(classifier, subtask, parameters):
    return 'python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c {} -subtask {} {} --data_version v3 -nfold 10 -hilbert >> $PRINTFILE'.format(
        classifier, subtask, parameters)


# python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c $c -subtask $subtask ${embedding} ${lda} ${fasttext} -gridsearchstrategy $gridsearchstrategy --data_version v3 -nfold 10 -hilbert >> $PRINTFILE

def append_header(handler, job_array_length):
    with open('template_header.txt', 'r') as content_file:
        content = content_file.read()
        content = content.replace("{JOBARRAY}", "#PBS -J 1-{}".format(job_array_length))
        handler.write(content)


if __name__ == '__main__':
    with open('hilbert_data_v3_jobarray.job', 'w') as handler:
        job_parameters = []
        job_parameters.extend(get_basic_features())
        job_parameters.extend(get_word_embedding_features())
        job_parameters.extend(get_character_embedding_features())
        job_parameters.extend(get_lda_features())
        append_header(handler, len(job_parameters) * len(subtasks) * len(classifiers))
        counter = 1
        for job_parameter in job_parameters:
            for subtask in subtasks:
                for classifier in classifiers:
                    handler.write("job_parameter[{}] = \"{}\"\n".format(counter,
                                                                         create_python_call(classifier, subtask,
                                                                                            job_parameter)))
                    counter += 1
