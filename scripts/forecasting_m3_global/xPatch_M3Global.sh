export CUDA_VISIBLE_DEVICES=0

model_name=xPatch

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
  --xpatch_stride 8 \
  --xpatch_padding end \
  --xpatch_alpha 0.3 \
  --xpatch_revin \
  --batch_size 128 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.0005 \
  --lradj type1 \
  --use_amp \
  --itr 1 \
  --des Exp
