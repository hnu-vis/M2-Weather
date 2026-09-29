export CUDA_VISIBLE_DEVICES=0

model_name=Corrformer

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
  --label_len 24 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --node_num 983 \
  --num_nodes 983 \
  --node_list 983 \
  --e_layers 2 \
  --d_layers 1 \
  --factor_temporal 1 \
  --factor_spatial 1 \
  --dec_tcn_layers 1 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --moving_avg 25 \
  --dropout 0.1 \
  --batch_size 1 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --itr 1 \
  --des Exp
