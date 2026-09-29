export CUDA_VISIBLE_DEVICES=0

model_name=STELLA

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
  --num_features 4 \
  --no_if_rel \
  --d_model 32 \
  --e_layers 2 \
  --dropout 0.2 \
  --batch_size 1024 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.0005 \
  --weight_decay 0.0005 \
  --lr_milestones 50 \
  --lr_gamma 0.5 \
  --clip_grad 5 \
  --itr 1 \
  --des Exp
