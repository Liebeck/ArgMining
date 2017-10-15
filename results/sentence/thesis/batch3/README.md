
# Jobs that need to be evaluated on a smaller parameter set:
python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c knn -subtask A -gridsearchstrategy character_ngrams -jobid 999 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE
3: job_parameter[3]="python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c knn -subtask A -gridsearchstrategy character_ngrams -jobid 3 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE"
4: job_parameter[4]="python -u /scratch_gs/malie102/jobs/ArgMining/scripts/sentence/gridsearch.py -c rf -subtask A -gridsearchstrategy character_ngrams -jobid 4 --data_version v3 -nfold 10 -hilbert >> $PRINTFILE"



## Unimportant Jobs that were killed due to the wall time being exceeded:

302: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-100
306: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-100
294: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-100
282:  svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-100
290: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-100
270: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-100 -jobid 270
278: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-100
266: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-100
258: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-100
254: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-100
206: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-50
182: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-20
150: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-20
62: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-5
42: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-5
102: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-10
222: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-50
186: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-20
218: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-50
198: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_3-50
246: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-50
30: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-5
90: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-10
234: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-50
230: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-4_4-50
66: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-5
162: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-20
210: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-50
146: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-3_6-20
242: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-50
158: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-5_5-20
126: svm-linear -subtask B -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-10
122: svm-linear -subtask A -gridsearchstrategy character_embeddings_centroid_100 -fasttext_path /scratch_gs/malie102/data/fasttext/dewiki-20170501-6_6-10