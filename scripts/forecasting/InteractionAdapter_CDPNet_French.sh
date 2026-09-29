export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python_bin=${PYTHON_BIN:-/root/miniconda3/envs/tslib/bin/python}

backbone_checkpoint=./experiment_records/forecasting_uvtrh/checkpoints/long_term_forecast_French_UVTRH_48_72_CDPNet_French_ftM_sl48_ll0_pl72_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_Exp_seed2024_0/checkpoint.pth

"$python_bin" -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/French/ \
  --data_path meteonet_nw_1h_2016_2018_nopsl_weather2k.npy \
  --station_coords_path ./dataset/French/station_coords.npy \
  --model_id French_UVTRH_48_72_CDPNet_VariableAdapter \
  --model InteractionAdapter \
  --backbone_model CDPNet \
  --backbone_checkpoint "$backbone_checkpoint" \
  --adapter_mode variable \
  --data French \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --num_nodes 133 \
  --num_node 133 \
  --feat_dim 32 \
  --grid_H 32 \
  --grid_W 32 \
  --grid_lat_min 46.0 \
  --grid_lat_max 51.25 \
  --grid_lon_min -5.5 \
  --grid_lon_max 2.5 \
  --n_neighbors 10 \
  --batch_size 384 \
  --train_epochs 30 \
  --resume \
  --patience 5 \
  --learning_rate 0.001 \
  --weight_decay 0.0001 \
  --itr 1 \
  --seed 2024 \
  --use_amp \
  --inverse \
  --checkpoints ./experiment_records/adapters/checkpoints/ \
  --results ./experiment_records/adapters/results/ \
  --test_results ./experiment_records/adapters/test_results/ \
  --result_file ./experiment_records/adapters/result_long_term_forecast.txt \
  --des Exp
