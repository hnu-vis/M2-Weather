export CUDA_VISIBLE_DEVICES=0

model_name=STELLA

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
  --num_features 4 \
  --no_if_rel \
  --d_model 32 \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 192 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.0005 \
  --weight_decay 0.0005 \
  --lr_milestones 50 \
  --lr_gamma 0.5 \
  --clip_grad 5 \
  --itr 1 \
  --des Exp
