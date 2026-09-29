export CUDA_VISIBLE_DEVICES=0

model_name=MIGN

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
  --mign_config ./scripts/forecasting/MIGN_French.yaml \
  --mign_graph_template ./dataset/French/MIGN/graph_48_72.bin \
  --mign_feature MXSPD \
  --batch_size 24 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.001 \
  --t_max 500 \
  --itr 1 \
  --des Exp
