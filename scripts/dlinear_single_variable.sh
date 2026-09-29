#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {u|v|T|RH}" >&2
  exit 2
fi

variable=$1
case "$variable" in
  u|v|T|RH) ;;
  *)
    echo "unsupported variable: $variable (expected u, v, T, or RH)" >&2
    exit 2
    ;;
esac

record_root=${RECORD_ROOT:-./experiment_records/forecasting_uvtrh_dlinear_single_matched}
seed=${SEED:-2024}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --inverse \
  --seed "$seed" \
  --root_path ./dataset/French/ \
  --data_path meteonet_nw_1h_2016_2018_nopsl_weather2k.npy \
  --model_id "French_${variable}_48_72" \
  --model DLinear \
  --data French \
  --features M \
  --state_variables "$variable" \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 72 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 2048 \
  --train_epochs 30 \
  --resume \
  --patience 10 \
  --learning_rate 0.0001 \
  --lradj type1 \
  --itr 1 \
  --checkpoints "$record_root/checkpoints" \
  --results "$record_root/results" \
  --test_results "$record_root/test_results" \
  --result_file "$record_root/result_long_term_forecast.txt" \
  --des Exp
