python task2_baseline.py --idx 0 --c1 0.5 --minfreq 0.4 --featstate 0 --maxiter 3000 --trans 0
python submission.py text.txt predictions_tags.txt
python task2_evaluate.py from-file --ref_file data/test_gold.csv task2_pred.csv