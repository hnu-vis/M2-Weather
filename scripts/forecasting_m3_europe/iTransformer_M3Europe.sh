export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/M3/Europe/ \
  --data_path m3_europe_europe_q0q1_weather5_1h_2017_2021.npy \
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
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --dropout 0.1 \
  --batch_size 24 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type1 \
  --itr 1 \
  --des Exp
