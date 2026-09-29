export CUDA_VISIBLE_DEVICES=0

model_name=DUET

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
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
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --d_model 256 \
  --n_heads 8 \
  --d_ff 512 \
  --patch_len 16 \
  --hidden_size 128 \
  --num_experts 4 \
  --k 1 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --aux_loss_weight 1.0 \
  --batch_size 256 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.02 \
  --lradj type3 \
  --itr 1 \
  --des Exp
