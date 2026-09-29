export CUDA_VISIBLE_DEVICES=0

model_name=Timer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/French/ \
  --data_path meteonet_nw_1h_2016_2018_nopsl_weather2k.npy \
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
  --timer_model_path thuml/timer-base-84m \
  --batch_size 128 \
  --num_workers 2 \
  --itr 1 \
  --des Exp
