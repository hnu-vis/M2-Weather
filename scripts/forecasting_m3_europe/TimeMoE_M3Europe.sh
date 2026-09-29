export CUDA_VISIBLE_DEVICES=0

model_name=TimeMoE

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
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
  --timemoe_model_path Maple728/TimeMoE-50M \
  --batch_size 1 \
  --num_workers 2 \
  --itr 1 \
  --des Exp
