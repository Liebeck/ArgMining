def extract_unfinished_jobs():
    job_ids = []
    with open('batchjob_detail') as file:
        for line in file:
            if line.startswith('269640') and 'R workq' in line:
                job_ids.append(int(line[line.find("[") + 1:line.find("]")]))
    return job_ids


def read_jobarray(unfinished_jobs):
    parameters_unfinished_jobs = []
    with open('hilbert_data_v3_jobarray.job') as file:
        for line in file:
            if line.startswith('job_parameter['):
                job_id = int(line[len('job_parameter['):line.find("]")])
                if job_id in unfinished_jobs:
                    # print(job_id)
                    parameters_raw = line[
                                     line.find('gridsearch.py ') + len('gridsearch.py '):line.find(" --data_version")]
                    parameters_split = parameters_raw.split(' ')
                    params = {'classifier': parameters_split[1],
                              'subtask': parameters_split[3],
                              'jobID': job_id,
                              'gridsearchstrategy': parameters_split[5]},
                    # 'other_params': parameters_split[6:]}
                    print(params)
                    parameters_unfinished_jobs.append(params)
                    # print(parameters_raw)
    return parameters_unfinished_jobs


if __name__ == '__main__':
    unfinished_jobs = extract_unfinished_jobs()
    parameters_unfinished_jobs = read_jobarray(unfinished_jobs)
