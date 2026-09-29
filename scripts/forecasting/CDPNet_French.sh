export CUDA_VISIBLE_DEVICES=0

model_name=CDPNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/French/ \
  --data_path meteonet_nw_1h_2016_2018_nopsl_weather2k.npy \
  --station_coords_path ./dataset/French/station_coords.npy \
  --model_id French_UVTRH_48_72 \
  --model $model_name \
  --data French \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --num_nodes 133 \
  --num_node 133 \
  --feat_dim 32 \
  --grid_H 32 \
  --grid_W 32 \
  --grid_lat_min 46.0 \
  --grid_lat_max 51.25 \
  --grid_lon_min -5.5 \
  --grid_lon_max 2.5 \
  --n_neighbors 10 \
  --batch_size 256 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --itr 1 \
  --des Exp
