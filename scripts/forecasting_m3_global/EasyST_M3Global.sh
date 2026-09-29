export CUDA_VISIBLE_DEVICES=0

model_name=EasyST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/M3/Global/ \
  --data_path m3_global_global_q0_weather5_1h_2017_2021.npy \
  --station_coords_path ./dataset/M3/Global/station_coords.npy \
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
  --num_nodes 2504 \
  --easyst_teacher_checkpoint auto \
  --easyst_embed_dim 64 \
  --easyst_node_dim 64 \
  --easyst_time_dim 64 \
  --easyst_transition_dim 64 \
  --easyst_num_layers 3 \
  --easyst_distill_weight 0.3 \
  --easyst_teacher_delta 0.1 \
  --easyst_ib_weight 0.001 \
  --dropout 0.1 \
  --batch_size 32 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.002 \
  --lradj type1 \
  --clip_grad 5 \
  --use_amp \
  --itr 1 \
  --des Exp
