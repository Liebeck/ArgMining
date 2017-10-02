import os


def append_header(handler, job_array_length):
    with open('template_header.txt', 'r') as content_file:
        content = content_file.read()
        content = content.replace("{JOBARRAY}", "#PBS -J 1-{}".format(job_array_length))
        handler.write(content)


def get_finished_and_unevaluated_jobs():
    evaluate_these_jobs = []
    basepath = '/scratch_gs/malie102/jobs/ArgMining/results/sentence/temp'
    completed_job_files = [f for f in os.listdir(basepath) if
                           f != '.gitignore' and not f.endswith('.predictions') and not f.endswith('.score')]
    all_files = [f for f in os.listdir(basepath) if f != '.gitignore']
    for completed_job in completed_job_files:
        score_file = '{}.score'.format(completed_job)
        if score_file not in all_files:
            evaluate_these_jobs.append(os.path.join(basepath, completed_job))
    return evaluate_these_jobs


def create_python_call(configfile_path):
    return 'python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/evaluate.py -configfile {} -hilbert'.format(
        configfile_path)


if __name__ == '__main__':
    with open('hilbert_evaluate_jobarray.job', 'w') as handler:
        job_parameters = get_finished_and_unevaluated_jobs()
        append_header(handler, len(job_parameters))
        counter = 1
        for job_parameter in job_parameters:
            handler.write("job_parameter[{}]=\"{}\"\n".format(counter,
                                                              create_python_call(job_parameter)))
            counter += 1
        handler.write("\n")
        handler.write("echo \"subjob: $PBS_ARRAY_INDEX\"\n")
        handler.write("eval ${job_parameter[$PBS_ARRAY_INDEX]}\n")
