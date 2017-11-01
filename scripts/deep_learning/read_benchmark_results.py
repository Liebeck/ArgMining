import os


def read_benchmark_results():
    finished_benchmarks = []
    result_path = 'results/sentence_deeplearning/temp'
    completed_benchmark_files = [f for f in os.listdir(result_path) if f.endswith('.score')]
    for file in completed_benchmark_files:
        split = file.split('_')
        subtask = split[0]
        if 'embedding_cnn_lstm' in file:
            keras_model_name = 'embedding_cnn_lstm'
            jobid = split[4]
        elif 'embedding_cnn' in file:
            keras_model_name = 'embedding_cnn'
            jobid = split[3]
        else:
            keras_model_name = split[1]
            jobid = split[2]
        type = split[-1].replace('.score', '')
        complete_path = os.path.join(result_path, file)
        with open(complete_path) as file_handler:
            f1_mean = 0
            f1_scores = []
            for line in file_handler:
                if line.startswith('Micro-averaged F1: '):
                    f1_mean = float(line[len('Micro-averaged F1: '):-1])
                if line.startswith('Individual scores: [ '):
                    f1_scores_raw = line[len('Individual scores: [ '):-2]
                    split = f1_scores_raw.split('  ')
                    for score in split:
                        if score and score != ' ':
                            f1_scores.append(float(score))
                            finished_benchmarks.append({'subtask': subtask,
                                                        'keras_model_name': keras_model_name,
                                                        'f1_mean': f1_mean,
                                                        'f1_scores': f1_scores,
                                                        'type': type,
                                                        'jobid': jobid})
                else:
                    continue
    return finished_benchmarks


def filter_benchmarks(finished_benchmarks, subtask, keras_model_name, type):
    filtered = []
    for result in finished_benchmarks:
        if result['subtask'] == subtask and result['keras_model_name'] == keras_model_name \
                and result['type'] == type:
            filtered.append(result)
    return filtered


if __name__ == '__main__':
    finished_benchmarks = read_benchmark_results()
    subtasks = ['A', 'B']
    keras_models = ['lstm-embedding-empty', 'lstm-embedding-pretrained', 'lstm-stacked', 'blstm', 'embedding_cnn',
                    'embedding_cnn_lstm']
    # types = ['last', 'best']
    types = ['last']
    for keras_model in keras_models:
        for subtask in subtasks:
            for type in types:
                filtered = filter_benchmarks(finished_benchmarks, subtask, keras_model, type)
                if filtered:
                    best_result = max(filtered, key=lambda d: d['f1_mean'])
                    print('Subtask: {}, Model: {}, Type: {}={:.2f}'.format(subtask,
                                                                           keras_model,
                                                                           type,
                                                                           best_result['f1_mean'] * 100))
                else:
                    print('Subtask: {}, Model: {}, Type: {} not run yet'.format(subtask,
                                                                                keras_model,
                                                                                type))
                    # print(len(filtered))
