python3 task2_baseline.py --idx 0 --c1 0.5 --minfreq 0.4 --featstate 0 --maxiter 3000 --trans 0
python3 submission.py text.txt predictions_tags.txt
python3 task2_evaluate.py from-file --ref_file data/train_test_gold.csv pred/task2_test.csv
# zip result.zip pred/task2.csv