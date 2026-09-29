export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/M3/Global/ \
  --data_path m3_global_global_q0_weather5_1h_2017_2021.npy \
  --model_id M3Global_TWSRHP_48_72 \
  --model $model_name \
  --data M3Global \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --patch_len 16 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --batch_size 8 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.0001 \
  --lradj TST \
  --pct_start 0.3 \
  --itr 1 \
  --des Exp
