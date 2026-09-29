export CUDA_VISIBLE_DEVICES=0

model_name=TimerXL

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
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
  --batch_size 64 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.000005 \
  --tmax 100 \
  --clip_grad 5 \
  --itr 1 \
  --des Exp
