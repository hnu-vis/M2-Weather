export CUDA_VISIBLE_DEVICES=0

model_name=Moirai

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/M3/France/ \
  --data_path m3_france_france_q0q1_weather5_1h_2017_2021.npy \
  --model_id M3France_TWSRHP_48_72 \
  --model $model_name \
  --data M3France \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --moirai_model_path Salesforce/moirai-2.0-R-small \
  --batch_size 8 \
  --num_workers 2 \
  --itr 1 \
  --des Exp
