import os


def read_all_score_files():
    finished_evaluations = []
    basepath = '/scratch_gs/malie102/jobs/ArgMining/results/sentence/temp'
    completed_evaluation_files = [f for f in os.listdir(basepath) if f.endswith('.score')]
    for file in completed_evaluation_files:
        stripped_file_name = file[0:-26]
        split = stripped_file_name.split('_')
        subtask = split[0]
        classifier = split[1]
        strategy = '_'.join(split[2:])
        complete_path = os.path.join(basepath, file)
        with open(complete_path) as file_handler:
            f1_mean = 0
            f1_scores = []
            for line in file_handler:
                if line.startswith('Micro-averaged F1: '):
                    f1_mean = line[len('Micro-averaged F1: '):-1]
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
                                                 'f1_scores': f1_scores})
                else:
                    continue
    return finished_evaluations

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



if __name__ == '__main__':
    finished_evaluations = read_all_score_files()
    subtask_A, subtask_B, subtask_C = group_results_by_subtask(finished_evaluations)
    print('Subtask A: {} jobs finished'.format(len(subtask_A)))
    print_sorted_results(subtask_A)
    print('Subtask B: {} jobs finished'.format(len(subtask_B)))
    print_sorted_results(subtask_B)
    print('Subtask C: {} jobs finished'.format(len(subtask_C)))
    print_sorted_results(subtask_C)



