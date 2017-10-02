import json
import os
import subprocess
import numpy as np


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
                          'other_params': parameters_split[7:-2]}
                parameters_all_jobs[job_id] = params
    return parameters_all_jobs


def get_qsub_jobid():
    '''
    Retrieves the jobid of the first job_array job from user 'malie102'
    '''
    process_output = subprocess.check_output(['qstat', '-u', 'malie102'])
    for line in process_output.split(b'\n'):
        if b'[]' in line:
            jobid = line[0:line.find(b"]") + 1]
            print('Jobid: {}'.format(jobid))
            return jobid


def get_finished_jobs(qsub_jobid):
    '''
    Retrieves all finished jobs with status 'X' from qstat -t JOBID
    '''
    finished_job_id = []
    process_output = subprocess.check_output(['qstat', '-t', qsub_jobid])
    line_prefix = qsub_jobid[:-1]
    for line in process_output.split(b'\n'):
        if line.startswith(bytes(line_prefix)) and b' X ' in line:
            sub_jobid = line[line.find(b"[") + 1:line.find(b"]")]
            finished_job_id.append(int(sub_jobid))
    return finished_job_id


def get_finished_jobs_json():
    '''
    Reads all JSON-serialized gridsearch results and retrieves the 'jobid' property
    '''
    finished_jobs_json_ids = []
    basepath = '/scratch_gs/malie102/jobs/ArgMining/results/sentence/temp'
    completed_job_files = [f for f in os.listdir(basepath) if
                           f != '.gitignore' and not f.endswith('.predictions') and not f.endswith('.score')]
    for json_file in completed_job_files:
        json_file_path = os.path.join(basepath, json_file)
        with open(json_file_path) as data_file:
            gridsearch_results = json.load(data_file)
            finished_jobs_json_ids.append(int(gridsearch_results['jobid']))
    return finished_jobs_json_ids


if __name__ == '__main__':
    all_planned_jobs = read_jobarray()
    qsub_jobid = get_qsub_jobid()
    finished_jobs = get_finished_jobs(qsub_jobid)
    finished_json_jobs = get_finished_jobs_json()
    jobs_with_errors = sorted(np.setdiff1d(finished_jobs, finished_json_jobs))
    print('{} jobs finished without writing a JSON file:'.format(len(jobs_with_errors)))
    for job in jobs_with_errors:
        print('Job {} did not finish'.format(job))
        job_dict = all_planned_jobs[job]
        del job_dict['jobID']
        print(job_dict)
        print()
