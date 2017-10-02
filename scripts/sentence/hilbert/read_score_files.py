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


if __name__ == '__main__':
    jobarray = read_jobarray()
    finished_evaluations = read_all_score_files(jobarray)
    subtask_A, subtask_B, subtask_C = group_results_by_subtask(finished_evaluations)
    print('Subtask A: {} jobs finished'.format(len(subtask_A)))
    print_sorted_results(subtask_A)
    print('Subtask B: {} jobs finished'.format(len(subtask_B)))
    print_sorted_results(subtask_B)
    print('Subtask C: {} jobs finished'.format(len(subtask_C)))
    print_sorted_results(subtask_C)
