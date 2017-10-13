All results containing character embeddings in batch2 are based on an incomplete lookup

JobIDs still running from batch v2 when v3 started:
1: job_parameter[1]="python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c svm -subtask A -gridsearchstrategy character_ngrams -jobid 1 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE"
3: job_parameter[3]="python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c knn -subtask A -gridsearchstrategy character_ngrams -jobid 3 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE"
4: job_parameter[4]="python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c rf -subtask A -gridsearchstrategy character_ngrams -jobid 4 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE"
7: job_parameter[7]="python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c knn -subtask B -gridsearchstrategy character_ngrams -jobid 7 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE"
8: job_parameter[8]="python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c rf -subtask B -gridsearchstrategy character_ngrams -jobid 8 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE"


