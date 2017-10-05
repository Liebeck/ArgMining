import os


def read_all_score_files(jobarray):
    finished_evaluations = []
    basepath = '/scratch_gs/malie102/jobs/ArgMining/results/sentence/temp'
    completed_evaluation_files = [f for f in os.listdir(basepath) if f.endswith('.score')]
    for file in completed_evaluation_files:
        stripped_file_name = file[0:-22]
        split = stripped_file_name.split('_')
        subtask = split[0]
        classifier = split[1]
        strategy = '_'.join(split[2:-1])
        jobid = split[-1]
        complete_path = os.path.join(basepath, file)
        with open(complete_path) as file_handler:
            f1_mean = 0
            f1_scores = []
            for line in file_handler:
                if line.startswith('Micro-averaged F1: '):
                    f1_mean = float(line[len('Micro-averaged F1: '):-1])
                if line.startswith('Individual scores: [ '):
                    f1_scores_raw = line[len('Individual scores: [ '):-2]
                    # print(f1_scores_raw)
                    split = f1_scores_raw.split('  ')
                    for score in split:
                        if score and score != ' ':
                            f1_scores.append(float(score))
                    # print(f1_scores)
                    finished_evaluations.append({'subtask': subtask,
                                                 'classifier': classifier,
                                                 'strategy': strategy,
                                                 'f1_mean': f1_mean,
                                                 'f1_scores': f1_scores,
                                                 'jobid': jobid,
                                                 'other_params': jobarray[int(jobid)]['other_params']})
                else:
                    continue
    return finished_evaluations


def read_jobarray():
    '''
    Reads all job definitions from the job array
    '''
    parameters_all_jobs = {}
    with open('/scratch_gs/malie102/jobs/ArgMining/scripts/sentence/hilbert/hilbert_data_v3_jobarray.job') as file:
        for line in file:
            if line.startswith('job_parameter['):
                job_id = int(line[len('job_parameter['):line.find("]")])
                parameters_raw = line[
                                 line.find('gridsearch.py ') + len('gridsearch.py '):line.find(" --data_version")]
                parameters_split = parameters_raw.split(' ')
                params = {'classifier': parameters_split[1],
                          'subtask': parameters_split[3],
                          'jobID': job_id,
                          'gridsearchstrategy': parameters_split[5],
                          'other_params': parameters_split[6:-2]}
                parameters_all_jobs[job_id] = params
    return parameters_all_jobs


def group_results_by_subtask(finished_evaluations):
    subtask_A = []
    subtask_B = []
    subtask_C = []
    for dic in finished_evaluations:
        if dic['subtask'] == 'A':
            subtask_A.append(dic)
        elif dic['subtask'] == 'B':
            subtask_B.append(dic)
        elif dic['subtask'] == 'C':
            subtask_C.append(dic)
    return subtask_A, subtask_B, subtask_C


def print_sorted_results(subtask):
    newlist = sorted(subtask, key=lambda k: k['f1_mean'], reverse=True)
    for entry in newlist:
        print(entry)
    print('\n\n\n\n\n\n')


def print_results_all_subtasks(subtask_A, subtask_B, subtask_C):
    print('Subtask A: {} jobs finished'.format(len(subtask_A)))
    print_sorted_results(subtask_A)
    print('Subtask B: {} jobs finished'.format(len(subtask_B)))
    print_sorted_results(subtask_B)
    print('Subtask C: {} jobs finished'.format(len(subtask_C)))
    print_sorted_results(subtask_C)


def filter_feature_and_classifier(subtask_results, feature, classifier):
    filtered = []
    for result in subtask_results:
        if result['classifier'] == classifier and result['strategy'] == feature:
            filtered.append(result)
    return filtered


def filter_by_lda_path(results, lda_path):
    filtered = []
    for result in results:
        if any(lda_path in s for s in result['other_params']):
            filtered.append(result)
    return filtered


def group_by_feature_and_classifier(subtask_results):
    result_dict = {}
    classifiers = ['svm', 'svm-linear', 'knn', 'rf']
    features = ['unigram', 'bigram', 'character_ngrams', 'grammatical_spacy', 'pos_distribution_spacy',
                'dependency_distribution_spacy', 'unigram+grammatical_spacy', 'n_unigram+shape', 'sentiws_distribution',
                'embedding_centroid_100', 'embedding_centroid_200', 'embedding_centroid_300',
                'character_embeddings_centroid_100', 'unigram+embedding_centroid_100',
                'unigram+embedding_centroid_200', 'unigram+embedding_centroid_300',
                'unigram+character_embeddings_centroid_100']
    for classifier in classifiers:
        for feature in features:
            key = '{}_{}'.format(classifier, feature)
            results_for_combination = filter_feature_and_classifier(subtask_results, feature, classifier)
            if results_for_combination:
                best_result = max(results_for_combination, key=lambda d: d['f1_mean'])
                result_dict[key] = best_result
            else:
                result_dict[key] = []
                # result_dict[key] = results_for_combination

    for classifier in classifiers:
        for feature in ['lda_distribution', 'unigram+lda_distribution']:
            for lda_path in ['thf', 'wikipedia']:
                results_for_combination = filter_feature_and_classifier(subtask_results, feature, classifier)
                results_for_combination = filter_by_lda_path(results_for_combination, lda_path)
                key = '{}_{}_{}'.format(classifier, feature, lda_path)
                if results_for_combination:
                    best_result = max(results_for_combination, key=lambda d: d['f1_mean'])
                    result_dict[key] = best_result
                else:
                    result_dict[key] = []
                    # result_dict[key] = results_for_combination
    return result_dict


def print_table_subtask(all_rows, best_subtask_1, best_subtask_2=None):
    classifiers = ['svm', 'svm-linear', 'knn', 'rf']
    header_row = [' '.ljust(40)]
    column_ljust = 7
    for classifier in classifiers:
        header_row.append(classifier.ljust(column_ljust))
    if best_subtask_2:
        for classifier in classifiers:
            header_row.append(classifier.ljust(column_ljust))
    print('\t& '.join(header_row) + '\\\\')
    for row in all_rows:
        columns = [row.ljust(40)]
        for classifier in classifiers:
            key = '{}_{}'.format(classifier, row)
            cell = "X".ljust(column_ljust)
            if best_subtask_1[key]:
                cell = "{:.2f}".format(best_subtask_1[key]['f1_mean'] * 100).ljust(column_ljust)
            columns.append(cell)
        if best_subtask_2:
            for classifier in classifiers:
                key = '{}_{}'.format(classifier, row)
                cell = "X".ljust(column_ljust)
                if best_subtask_2[key]:
                    cell = "{:.2f}".format(best_subtask_2[key]['f1_mean'] * 100).ljust(column_ljust)
                columns.append(cell)
        print('\t& '.join(columns) + '\\\\')


def print_tables(subtask_A, subtask_B, subtask_C):
    best_subtask_A = group_by_feature_and_classifier(subtask_A)
    best_subtask_B = group_by_feature_and_classifier(subtask_B)
    best_subtask_C = group_by_feature_and_classifier(subtask_C)
    all_rows = ['unigram', 'bigram', 'pos_distribution_spacy', 'dependency_distribution_spacy',
                'grammatical_spacy',
                'n_unigram+shape', 'lda_distribution_wikipedia', 'lda_distribution_thf', 'character_ngrams',
                'embedding_centroid_100', 'embedding_centroid_200', 'embedding_centroid_300', 'sentiws_distribution',
                'character_embeddings_centroid_100', 'unigram+grammatical_spacy', 'unigram+lda_distribution_thf',
                'unigram+lda_distribution_wikipedia', 'unigram+embedding_centroid_100',
                'unigram+embedding_centroid_200', 'unigram+embedding_centroid_300',
                'unigram+character_embeddings_centroid_100']
    print('*****\nSubtask A+B')
    print_table_subtask(all_rows, best_subtask_A, best_subtask_B)
    print('*****\nSubtask C')
    print_table_subtask(all_rows, best_subtask_C)


def print_fasttext_results(finished_evaluations):
    print('Fasttext results')
    subtasks = ['A', 'B', 'C']
    classifiers = ['svm', 'svm-linear', 'knn', 'rf']
    ngram_sizes = ['3_3', '4_4', '5_5', '6_6', '3_6']

    for subtask in subtasks:
        for classifier in classifiers:
            for ngram_size in ngram_sizes:
                results = filter_fasttext_results(finished_evaluations, subtask, classifier, ngram_size)
                if len(results) == 5:
                    best_f1 = results[0]['f1_mean']
                    worst_f1 = results[-1]['f1_mean']
                    scores = []
                    for result in results:
                        scores.append((int(result['other_params'][2].split('-')[-1]), result['f1_mean']))
                    scores = sorted(scores, key=lambda k: k[0])
                    print(
                        'Subtask {}, Classifier {}, Ngram {} : {}, {}, {}'.format(subtask, classifier, ngram_size,
                                                                                  best_f1,
                                                                                  worst_f1, scores))
                else:
                    print(
                        'Subtask {}, Classifier {}, Ngram {} : Only {} evaluations yet'.format(subtask, classifier,
                                                                                               ngram_size,
                                                                                               len(results)))


def print_fasttext_results_latex(finished_evaluations):
    print('Fasttext results')
    subtasks = ['A', 'B']
    classifiers = ['svm', 'rf', 'knn']
    ngram_sizes = ['3_3', '4_4', '5_5', '6_6', '3_6']
    for subtask in subtasks:
        for classifier in classifiers:
            for ngram_size in ngram_sizes:
                results = filter_fasttext_results(finished_evaluations, subtask, classifier, ngram_size)
                best_f1 = results[0]['f1_mean']
                worst_f1 = results[-1]['f1_mean']
                scores = []
                for result in results:
                    scores.append((int(result['other_params'][2].split('-')[-1]), result['f1_mean']))
                scores = sorted(scores, key=lambda k: k[0])
                print(
                    'Subtask {}, Classifier {}, Ngram {} : {:.2f}, {}, {}'.format(subtask, classifier, ngram_size,
                                                                                  best_f1 * 100,
                                                                                  worst_f1, scores))


def filter_fasttext_results(finished_evaluations, subtask, classifier, ngram_size):
    filtered = []
    for result in finished_evaluations:
        if result['subtask'] == subtask and result['classifier'] == classifier and result[
            'strategy'] == 'character_embeddings_centroid_100' and ngram_size in result['other_params'][2]:
            filtered.append(result)
    filtered = sorted(filtered, key=lambda k: k['f1_mean'], reverse=True)
    return filtered


if __name__ == '__main__':
    jobarray = read_jobarray()
    finished_evaluations = read_all_score_files(jobarray)
    subtask_A, subtask_B, subtask_C = group_results_by_subtask(finished_evaluations)
    print_tables(subtask_A, subtask_B, subtask_C)
    # print('\n\n\n\n')
    # print_fasttext_results(finished_evaluations)
    # print_results_all_subtasks(subtask_A, subtask_B, subtask_C)
    # print_fasttext_results_latex(finished_evaluations)
