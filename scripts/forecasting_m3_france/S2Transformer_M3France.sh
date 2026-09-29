export CUDA_VISIBLE_DEVICES=0

model_name=S2Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/M3/France/ \
  --data_path m3_france_france_q0q1_weather5_1h_2017_2021.npy \
  --station_coords_path ./dataset/M3/France/station_coords.npy \
  --model_id M3France_TWSRHP_48_72 \
  --model $model_name \
  --data M3France \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --num_nodes 176 \
  --s2_group_counts 16 4 \
  --e_layers 2 \
  --n_heads 8 \
  --d_model 256 \
  --d_ff 512 \
  --dropout 0.1 \
  --batch_size 384 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --use_amp \
  --weight_decay 0.0001 \
  --clip_grad 5 \
  --itr 1 \
  --des Exp
