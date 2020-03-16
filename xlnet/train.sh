python run_classifier.py \
  --do_train=True \
  --do_eval=True \
  --task_name=pdtb \
  --data_dir=../../../nfs/nas-7.1/pwgao/data/PDTB-3.0/ \
  --output_dir=proc_data/pdtb \
  --model_dir=exp/pdtb \
  --uncased=False \
  --spiece_model_file=../../../nfs/nas-7.1/pwgao/data/xlnet_cased_L-12_H-768_A-12/spiece.model \
  --model_config_path=../../../nfs/nas-7.1/pwgao/data/xlnet_cased_L-12_H-768_A-12/model_config.json \
  --init_checkpoint=../../../nfs/nas-7.1/pwgao/data/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --eval_batch_size=8
  --num_hosts=1 \
  --num_core_per_host=4 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500