import argparse
import os
import time
import torch
import sys
import torch.distributed as dist
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selected Time-Series Baselines')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, choices=['long_term_forecast'])
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name; see MODEL_CATALOG.md for the 16 supported long-term entries')
    parser.add_argument('--seed', type=int, default=2026, help="Randomization seed")
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='prefer deterministic CUDA kernels for reproducible runs')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--station_coords_path', type=str, default=None,
                        help='station coordinate file [N,3]; defaults to <root_path>/station_coords.npy')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument(
        '--state_variables', nargs='+', choices=['u', 'v', 'T', 'WS', 'RH', 'P'], default=None,
        help=(
            'optional state subset; dataset defaults are French=[u,v,T,RH] and '
            'M3France/M3Europe/M3Global=[T,WS,RH,P]'
        ),
    )
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=None, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=None, help='start token length')
    parser.add_argument('--pred_len', type=int, default=None, help='prediction sequence length')
    parser.add_argument('--results', type=str, default='./results/', help='location of numeric test results')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test visualizations')
    parser.add_argument('--result_file', type=str, default='./result_long_term_forecast.txt', help='aggregate metric log')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)


    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default=None, help='activation (model-specific default when omitted)')
    parser.add_argument('--output_attention', action='store_true', default=False, help='return attention maps when supported')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument(
        '--train_stride', type=int, default=1,
        help='subsample training forecast origins only; validation/test remain complete',
    )
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=None, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size of train input data')
    parser.add_argument(
        '--eval_batch_size', type=int, default=None,
        help='validation/test batch size; defaults to --batch_size',
    )
    parser.add_argument('--patience', type=int, default=None, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=None, help='optimizer learning rate (model model-specific default when omitted)')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default=None, help='adjust learning rate (model model-specific default when omitted)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument(
        '--resume', action='store_true', default=False,
        help='resume from last_checkpoint.pth, or warm-start from a legacy checkpoint.pth',
    )
    parser.add_argument(
        '--skip_epoch_test', action='store_true', default=False,
        help='skip test-set evaluation inside each epoch; final test still runs',
    )
    parser.add_argument(
        '--skip_final_test', action='store_true', default=False,
        help='skip the post-training test pass for a controlled follow-up evaluation',
    )
    parser.add_argument('--weight_decay', type=float, default=None, help='override model optimizer weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, help='override model gradient clipping norm')
    parser.add_argument(
        '--profile_steps', type=int, default=0,
        help='run this many representative batches, report timing/memory, and exit',
    )
    parser.add_argument(
        '--profile_warmup_steps', type=int, default=0,
        help='run this many unmeasured batches before profiling',
    )


    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument("--use_ddp", action="store_true", default=False,
                        help="use torchrun DistributedDataParallel")
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # model-specific hyperparameters
    dlinear_group = parser.add_argument_group('DLinear')
    dlinear_group.add_argument('--individual', action='store_true', default=False,
                               help='a linear layer for each variate(channel) individually')

    patch_tqnet_group = parser.add_argument_group('PatchTST / TQNet')
    patch_tqnet_group.add_argument('--patch_len', type=int, default=16,
                                   help='PatchTST patch length')
    patch_tqnet_group.add_argument('--pct_start', type=float, default=0.3,
                                   help='OneCycle warm-up fraction')

    xpatch_group = parser.add_argument_group('xPatch')
    xpatch_group.add_argument('--xpatch_stride', type=int, default=8,
                              help='stride between xPatch temporal patches')
    xpatch_group.add_argument('--xpatch_padding', choices=['none', 'end'], default='end',
                              help='optional end replication padding')
    xpatch_group.add_argument('--xpatch_alpha', type=float, default=0.3,
                              help='EMA smoothing coefficient')
    xpatch_group.add_argument('--xpatch_revin', action='store_true', default=True,
                              help='enable xPatch reversible instance normalization')
    xpatch_group.add_argument('--no_xpatch_revin', action='store_false',
                              dest='xpatch_revin')

    s2_group = parser.add_argument_group('S2Transformer')
    s2_group.add_argument('--s2_group_counts', type=int, nargs='+', default=[16, 4],
                          help='fine-to-coarse geographic subgraph counts')

    easyst_group = parser.add_argument_group('EasyST')
    easyst_group.add_argument('--easyst_teacher_checkpoint', type=str, default='auto',
                              help='frozen STELLA teacher checkpoint, or auto')
    easyst_group.add_argument('--easyst_embed_dim', type=int, default=64)
    easyst_group.add_argument('--easyst_node_dim', type=int, default=64)
    easyst_group.add_argument('--easyst_time_dim', type=int, default=64)
    easyst_group.add_argument('--easyst_transition_dim', type=int, default=64)
    easyst_group.add_argument('--easyst_num_layers', type=int, default=3)
    easyst_group.add_argument('--easyst_distill_weight', type=float, default=0.3)
    easyst_group.add_argument('--easyst_teacher_delta', type=float, default=0.1)
    easyst_group.add_argument('--easyst_ib_weight', type=float, default=0.001)

    duet_group = parser.add_argument_group('DUET')
    duet_group.add_argument('--fc_dropout', type=float, default=0.2, help='output-head dropout')
    duet_group.add_argument('--hidden_size', type=int, default=256, help='temporal router width')
    duet_group.add_argument('--num_experts', type=int, default=4, help='number of temporal experts')
    duet_group.add_argument('--k', type=int, default=1, help='number of active experts')
    duet_group.add_argument('--noisy_gating', action='store_true', default=True,
                            help='use noisy expert routing')
    duet_group.add_argument('--CI', action='store_true', default=True,
                            help='use the channel-independent temporal extractor')
    duet_group.add_argument('--aux_loss_weight', type=float, default=1.0,
                            help='load-balancing auxiliary-loss weight')

    tqnet_group = parser.add_argument_group('TQNet')
    tqnet_group.add_argument('--cycle', type=int, default=24, help='cycle length')
    tqnet_group.add_argument('--cycle_mark_column', type=int, default=2,
                             help='time-mark column used as phase')
    tqnet_group.add_argument('--cycle_offset', type=int, default=0,
                             help='offset added to the cycle phase')
    tqnet_group.add_argument('--model_type', type=str, default='mlp', help='model variant')
    tqnet_group.add_argument('--use_revin', action='store_true', default=True,
                             help='use the model RevIN path')

    corrformer_group = parser.add_argument_group('Corrformer')
    corrformer_group.add_argument('--node_num', type=int, default=1, help='station count')
    corrformer_group.add_argument('--node_list', type=int, nargs='+', default=[1],
                                  help='hierarchy factors')
    corrformer_group.add_argument('--factor_temporal', type=int, default=1,
                                  help='temporal correlation factor')
    corrformer_group.add_argument('--factor_spatial', type=int, default=1,
                                  help='spatial correlation factor')
    corrformer_group.add_argument('--dec_tcn_layers', type=int, default=1,
                                  help='causal-convolution depth')

    cdpnet_group = parser.add_argument_group('CDPNet')
    cdpnet_group.add_argument('--num_node', type=int, default=None, help='station count')
    cdpnet_group.add_argument('--feat_in', type=int, default=None,
                              help='predicted features per station')
    cdpnet_group.add_argument('--input_dim', type=int, default=None,
                              help='per-station embedding input dimension')
    cdpnet_group.add_argument('--feat_dim', type=int, default=32, help='hidden feature dimension')
    cdpnet_group.add_argument('--grid_H', type=int, default=None, help='interpolation grid height')
    cdpnet_group.add_argument('--grid_W', type=int, default=None, help='interpolation grid width')
    cdpnet_group.add_argument('--n_neighbors', type=int, default=None,
                              help='number of interpolation neighbors')
    cdpnet_group.add_argument('--grid_lat_min', type=float, default=None)
    cdpnet_group.add_argument('--grid_lat_max', type=float, default=None)
    cdpnet_group.add_argument('--grid_lon_min', type=float, default=None)
    cdpnet_group.add_argument('--grid_lon_max', type=float, default=None)

    spatial_group = parser.add_argument_group('STELLA / HiSTGNN')
    spatial_group.add_argument('--num_nodes', type=int, default=1, help='station count')
    spatial_group.add_argument('--num_features', type=int, default=1,
                               help='variables per station')

    stella_group = parser.add_argument_group('STELLA')
    stella_group.add_argument('--if_rel', action='store_true', default=True,
                              help='use relative positional encoding')
    stella_group.add_argument('--no_if_rel', action='store_false', dest='if_rel',
                              help='use the provided station coordinates')
    stella_group.add_argument('--res_conn', action='store_true', default=True,
                              help='use the residual MLP connection')
    stella_group.add_argument('--lr_milestones', type=int, nargs='+', default=None,
                              help='MultiStepLR milestones')
    stella_group.add_argument('--lr_gamma', type=float, default=0.5, help='MultiStepLR gamma')

    histgnn_group = parser.add_argument_group('HiSTGNN')
    histgnn_group.add_argument('--gcn_depth', type=int, default=2, help='GCN depth')
    histgnn_group.add_argument('--propalpha', type=float, default=0.3,
                               help='graph propagation retention ratio')
    histgnn_group.add_argument('--conv_channel', type=int, default=32,
                               help='graph convolution channels')
    histgnn_group.add_argument(
        '--histgnn_vectorized', action='store_true', default=True,
        help='batch station-local HiSTGNN operations (default: on)',
    )
    histgnn_group.add_argument(
        '--no_histgnn_vectorized', action='store_false',
        dest='histgnn_vectorized', help='use the reference station loop',
    )

    mign_group = parser.add_argument_group('MIGN')
    mign_group.add_argument('--mign_config', type=str, default=None, help='YAML configuration')
    mign_group.add_argument('--mign_feature', type=str, default='MXSPD',
                            help='target graph node type')
    mign_group.add_argument('--mign_data_dir', type=str, default=None,
                            help='override the YAML data_dir')
    mign_group.add_argument('--mign_graph_template', type=str, default=None,
                            help='static DGL .bin graph used to adapt dense station tensors')
    mign_group.add_argument('--mign_realtime', action='store_true', default=False,
                            help='use the real-time DGL dataset')
    mign_group.add_argument('--t_max', type=int, default=500, help='CosineAnnealingLR T_max')

    timer_group = parser.add_argument_group('Timer')
    timer_group.add_argument('--timer_model_path', type=str, default='thuml/timer-base-84m',
                             help='Hugging Face model ID or local snapshot')

    sundial_group = parser.add_argument_group('Sundial')
    sundial_group.add_argument('--sundial_model_path', type=str,
                               default='thuml/sundial-base-128m')

    timemoe_group = parser.add_argument_group('TimeMoE')
    timemoe_group.add_argument('--timemoe_model_path', type=str,
                               default='Maple728/TimeMoE-50M')
    parser.add_argument(
        '--foundation_chunk_size', type=int, default=0,
        help='independent-series inference chunk; 0 uses each foundation model default',
    )

    moirai_group = parser.add_argument_group('Moirai')
    moirai_group.add_argument('--moirai_model_path', type=str,
                              default='Salesforce/moirai-2.0-R-small')

    timerxl_group = parser.add_argument_group('TimerXL')
    timerxl_group.add_argument('--timerxl_model_path', type=str, default=None,
                               help='OpenLTM .pth checkpoint; defaults to pretrained_checkpoints/Timer-XL/checkpoint.pth')
    timerxl_group.add_argument('--input_token_len', type=int, default=96, help='input token length')
    timerxl_group.add_argument('--output_token_len', type=int, default=96, help='output token length')
    timerxl_group.add_argument('--covariate', action='store_true', default=False,
                               help='use the covariate attention mask')
    timerxl_group.add_argument('--flash_attention', action='store_true', default=False,
                               help='use the flash-attention path')
    timerxl_group.add_argument('--use_norm', action='store_true', default=True,
                               help='use instance normalization')
    timerxl_group.add_argument('--no_use_norm', action='store_false', dest='use_norm',
                               help='disable instance normalization')
    timerxl_group.add_argument('--cosine', action='store_true', default=True,
                               help='use the OpenLTM cosine scheduler (default: on)')
    timerxl_group.add_argument('--no_cosine', action='store_false', dest='cosine',
                               help='disable the OpenLTM cosine scheduler')
    timerxl_group.add_argument('--tmax', type=int, default=None, help='CosineAnnealingLR T_max; defaults to train_epochs')

    # optional hook shared by spatial/graph data pipelines
    parser.add_argument('--custom_data_provider', type=str, default=None,
                        help='module:function returning an (dataset, loader)')
    parser.add_argument(
        '--m3_split_profile', choices=['default', 'adapter_sensitivity'],
        default='default',
        help='chronological split policy for M3 experiments',
    )

    adapter_group = parser.add_argument_group('InteractionAdapter')
    adapter_group.add_argument('--backbone_model', type=str, default=None,
                               help='pretrained model wrapped by the adapter')
    adapter_group.add_argument('--backbone_checkpoint', type=str, default=None,
                               help='best or resumable checkpoint of the frozen backbone')
    adapter_group.add_argument('--adapter_mode', choices=['variable', 'station', 'both'],
                               default='both', help='interaction branch(es) to train')
    adapter_group.add_argument('--adapter_k_neighbors', type=int, default=8,
                               help='geographic neighbors used by the station branch')
    adapter_group.add_argument('--adapter_calibration', action='store_true', default=True,
                               help='learn lead/variable residual calibration (default: on)')
    adapter_group.add_argument('--no_adapter_calibration', action='store_false',
                               dest='adapter_calibration', help='disable calibration ablation')
    adapter_group.add_argument('--adapter_closed_form', action='store_true', default=False,
                               help='fit the frozen-backbone linear adapter from streaming sufficient statistics')
    adapter_group.add_argument(
        '--adapter_knn_sensitivity', action='store_true', default=False,
        help='fit several geographic kNN Adapter graphs while reusing each backbone prediction',
    )
    adapter_group.add_argument(
        '--adapter_knn_values', type=int, nargs='+', default=[2, 4, 8, 16],
        help='neighbor counts evaluated by --adapter_knn_sensitivity',
    )
    adapter_group.add_argument('--adapter_fit_stride', type=int, default=1,
                               help='window stride used for adapter train/validation fitting')
    adapter_group.add_argument('--adapter_ridge', type=float, nargs='+', default=None,
                               help='ridge candidates selected by validation MSE')
    adapter_group.add_argument(
        '--adapter_source_sensitivity', action='store_true', default=False,
        help='compare train-residual and held-out-residual closed-form fits',
    )
    args = parser.parse_args()
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.is_main_process = True
    if args.use_ddp:
        if not args.use_gpu or args.gpu_type != "cuda" or not torch.cuda.is_available():
            parser.error("--use_ddp requires CUDA and torchrun")
        dist.init_process_group(backend="nccl")
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.is_main_process = args.rank == 0
        args.gpu = args.local_rank
        args.use_multi_gpu = False
        torch.cuda.set_device(args.local_rank)
        if not args.is_main_process:
            sys.stdout = open(os.devnull, "w")
    ddp_enabled = args.use_ddp
    if args.train_stride <= 0:
        parser.error('--train_stride must be a positive integer')
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error('--eval_batch_size must be a positive integer')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # Fixed-shape forecasting batches benefit from TensorFloat-32 and
        # cuDNN autotuning on Ampere-or-newer GPUs without changing tensor
        # shapes, checkpoints, or the installed PyTorch environment.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = args.deterministic
        if args.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        if args.model == "TimeMoE":
            # Match closed-form Adapter inference, which disables matmul TF32.
            # Shared test predictions must use the same numerical settings.
            torch.backends.cuda.matmul.allow_tf32 = False
    if args.activation is None:
        args.activation = "relu" if args.model == "TimerXL" else "gelu"
    inference_only_models = {"Moirai", "Sundial", "TimeMoE", "Timer"}
    # TimerXL initializes from its public pretrained weights, but unlike the
    # inference-only foundation models it is subsequently fine-tuned by this
    # project.  Test-only runs must therefore reload the experiment checkpoint.
    self_loading_inference_models = inference_only_models
    if args.is_training and args.model in inference_only_models:
        parser.error(f"{args.model} is inference-only; use --is_training 0.")
    if args.use_gpu and args.use_multi_gpu and not args.use_ddp:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    Exp = Exp_Long_Term_Forecast

    if args.adapter_source_sensitivity:
        from utils.adapter_source_sensitivity import run_source_sensitivity
        run_source_sensitivity(args, Exp)
        raise SystemExit(0)
    if args.adapter_knn_sensitivity:
        from utils.adapter_knn_sensitivity import run_knn_sensitivity
        run_knn_sensitivity(args, Exp)
        raise SystemExit(0)
    if args.adapter_closed_form:
        from utils.fit_interaction_adapter import fit_interaction_adapter
        fit_interaction_adapter(args, Exp)
        raise SystemExit(0)

    def make_setting(iteration):
        return "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_seed{}_{}".format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed,
            args.des, args.seed, iteration)

    if args.profile_steps > 0:
        exp = Exp(args)
        profile_flag = "train" if args.is_training else "test"
        profile_data, profile_loader = exp._get_data(flag=profile_flag)
        criterion = exp._select_criterion()
        optimizer = exp._select_optimizer() if args.is_training else None
        profile_use_amp = (
            args.use_amp and args.use_gpu and args.gpu_type == "cuda"
        )
        profile_scaler = torch.cuda.amp.GradScaler(enabled=profile_use_amp)
        exp.model.train(bool(args.is_training))
        completed = 0
        measured = 0
        started = None
        total_profile_steps = args.profile_warmup_steps + args.profile_steps
        for completed, batch in enumerate(profile_loader, start=1):
            if completed == args.profile_warmup_steps + 1:
                if args.use_gpu and args.gpu_type == "cuda":
                    torch.cuda.synchronize(exp.device)
                    torch.cuda.reset_peak_memory_stats(exp.device)
                started = time.perf_counter()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(optimizer is not None):
                with torch.cuda.amp.autocast(enabled=profile_use_amp):
                    _, _, loss = exp._forward_batch(
                        batch, criterion, profile_data,
                        epoch=1 if args.is_training else None,
                    )
            if optimizer is not None:
                if profile_use_amp:
                    profile_scaler.scale(loss).backward()
                    profile_scaler.unscale_(optimizer)
                    exp.training_strategy.clip(exp.model, args)
                    profile_scaler.step(optimizer)
                    profile_scaler.update()
                else:
                    loss.backward()
                    exp.training_strategy.clip(exp.model, args)
                    optimizer.step()
            if completed > args.profile_warmup_steps:
                measured += 1
            if completed >= total_profile_steps:
                break
        if started is None:
            raise RuntimeError("Profile loader ended during warm-up; no measured steps.")
        if args.use_gpu and args.gpu_type == "cuda":
            torch.cuda.synchronize(exp.device)
            peak_memory_mb = torch.cuda.max_memory_allocated(exp.device) / 2 ** 20
            peak_reserved_mb = torch.cuda.max_memory_reserved(exp.device) / 2 ** 20
        else:
            peak_memory_mb = 0.0
            peak_reserved_mb = 0.0
        elapsed = time.perf_counter() - started
        print(
            f"PROFILE_RESULT model={args.model} "
            f"batch_size={int(batch[0].shape[0])} "
            f"warmup_steps={min(args.profile_warmup_steps, completed)} "
            f"steps={measured} elapsed_seconds={elapsed:.6f} "
            f"seconds_per_step={elapsed / max(measured, 1):.6f} "
            f"peak_memory_mb={peak_memory_mb:.1f} "
            f"peak_reserved_mb={peak_reserved_mb:.1f}",
            flush=True,
        )
        if ddp_enabled:
            dist.barrier()
            dist.destroy_process_group()
        raise SystemExit(0)

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = make_setting(ii)
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)
            if ddp_enabled:
                dist.barrier()
            if args.is_main_process and not args.skip_final_test:
                if ddp_enabled:
                    exp.model = exp._model_without_parallel()
                    exp.args.use_ddp = False
                print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                exp.test(setting)
            if ddp_enabled:
                dist.barrier()
            if args.use_gpu:
                if args.gpu_type == "mps":
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == "cuda":
                    torch.cuda.empty_cache()
        if ddp_enabled:
            dist.destroy_process_group()
    else:
        exp = Exp(args)
        setting = make_setting(0)
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=0 if args.model in self_loading_inference_models else 1)
        if args.use_gpu:
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
