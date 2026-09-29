export CUDA_VISIBLE_DEVICES=0

model_name=TQNet

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
  --cycle 24 \
  --model_type mlp \
  --patch_len 16 \
  --dropout 0.5 \
  --batch_size 1024 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.001 \
  --lradj TST \
  --pct_start 0.3 \
  --itr 1 \
  --seed 2024 \
  --des Exp
