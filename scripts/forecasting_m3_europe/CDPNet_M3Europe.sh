export CUDA_VISIBLE_DEVICES=0

model_name=CDPNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/M3/Europe/ \
  --data_path m3_europe_europe_q0q1_weather5_1h_2017_2021.npy \
  --station_coords_path ./dataset/M3/Europe/station_coords.npy \
  --model_id M3Europe_TWSRHP_48_72 \
  --model $model_name \
  --data M3Europe \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --num_nodes 983 \
  --num_node 983 \
  --feat_dim 32 \
  --grid_H 32 \
  --grid_W 32 \
  --n_neighbors 10 \
  --batch_size 48 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --itr 1 \
  --des Exp
