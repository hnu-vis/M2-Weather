export CUDA_VISIBLE_DEVICES=0

model_name=HiSTGNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/M3/Global/ \
  --data_path m3_global_global_q0_weather5_1h_2017_2021.npy \
  --model_id M3Global_TWSRHP_48_72 \
  --model $model_name \
  --data M3Global \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --num_nodes 2504 \
  --num_features 4 \
  --gcn_depth 2 \
  --conv_channel 32 \
  --propalpha 0.05 \
  --dropout 0.3 \
  --batch_size 10 \
  --eval_batch_size 21 \
  --use_amp \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --clip_grad 5 \
  --itr 1 \
  --des Exp
