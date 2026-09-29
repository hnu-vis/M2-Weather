from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.training_strategies import get_training_strategy
from utils.tools import EarlyStopping, visual
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import time
import warnings
import numpy as np
from utils.dtw_metric import accelerated_dtw

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        self.training_strategy = get_training_strategy(args.model)
        super(Exp_Long_Term_Forecast, self).__init__(args)
        if args.model == "MIGN":
            model = self._model_without_parallel()
            self.args.t_max = model.model_args.get("t_max", self.args.t_max)
            self.args.seq_len = model.input_length
            self.args.pred_len = model.output_length

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if getattr(self.args, "use_ddp", False):
            model = model.to(self.device)
            model = DDP(
                model, device_ids=[self.args.local_rank],
                output_device=self.args.local_rank, static_graph=True,
                gradient_as_bucket_view=True, bucket_cap_mb=100,
            )
        elif self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return self.training_strategy.make_optimizer(self.model, self.args)

    def _select_criterion(self):
        return self.training_strategy.make_criterion()

    def _with_auxiliary_loss(self, loss, prediction=None, target=None):
        unwrapped = self._model_without_parallel()
        objective = getattr(unwrapped, "training_objective", None)
        if objective is not None and self.model.training:
            return objective(loss, prediction, target)
        return self.training_strategy.add_auxiliary_loss(loss, self.model, self.args)
    def _model_without_parallel(self):
        return self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model

    def _forward_batch(self, batch, criterion, data_set, epoch=0):
        """Return prediction, target and configured objective for one upstream batch."""
        name = self.args.model

        if not isinstance(batch, (tuple, list)) or len(batch) != 5:
            raise TypeError(f"{name} expects the canonical five-field forecasting batch.")
        batch_x, batch_y, batch_x_mark, batch_y_mark, forecast_start = batch
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        batch_y = batch_y.float().to(self.device, non_blocking=True)
        batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
        batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
        forecast_start = forecast_start.to(
            self.device, dtype=torch.long, non_blocking=True
        )
        timerxl_training_batch = name == "TimerXL" and epoch is not None
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, ...])
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, ...], dec_inp], dim=1)

        if name == "CDPNet":
            self._model_without_parallel().training_epoch = epoch
        if timerxl_training_batch:
            outputs = self._model_without_parallel().forecast_training(
                batch_x, batch_x_mark, batch_y_mark
            )
        elif name in {"TQNet", "EasyST", "InteractionAdapter"}:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, forecast_start
            )
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if name == "CDPNet":
            target = batch_y if epoch is None or epoch == 0 else torch.cat([batch_x, batch_y], dim=1)[:, 1:]
            prediction = outputs
            if prediction.ndim == 3:
                prediction = prediction.reshape(
                    prediction.shape[0], prediction.shape[1], self.args.num_node, -1
                )
            if target.ndim == 3:
                target = target.reshape(target.shape[0], target.shape[1], self.args.num_node, -1)
            loss = criterion(prediction, target)
            return prediction.flatten(start_dim=2), target.flatten(start_dim=2), loss

        f_dim = -1 if self.args.features == "MS" else 0
        if timerxl_training_batch:
            prediction = outputs
            target = batch_y
            return prediction, target, criterion(prediction, target)
        prediction = outputs[:, -self.args.pred_len:, ...]
        target = batch_y[:, -self.args.pred_len:, ...]
        if prediction.ndim == 4:
            prediction = prediction.flatten(start_dim=2)
        if target.ndim == 4:
            target = target.flatten(start_dim=2)
        prediction = prediction[..., f_dim:]
        target = target[..., f_dim:]
        loss = self._with_auxiliary_loss(
            criterion(prediction, target), prediction, target
        )
        return prediction, target, loss

    def vali(self, vali_data, vali_loader, criterion, epoch=0):
        weighted_loss = 0.0
        examples = 0
        self.model.eval()
        use_amp = getattr(self.args, "use_amp", False) and self.device.type == "cuda"
        with torch.no_grad():
            for batch in vali_loader:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, _, loss = self._forward_batch(
                        batch, criterion, vali_data, epoch=epoch
                    )
                batch_size = int(batch[0].shape[0])
                weighted_loss += loss.item() * batch_size
                examples += batch_size
        if getattr(self.args, "use_ddp", False):
            totals = torch.tensor(
                [weighted_loss, examples], dtype=torch.float64, device=self.device
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            weighted_loss, examples = totals.tolist()
        self.model.train()
        return weighted_loss / max(examples, 1)

    @staticmethod
    def _early_stopping_state(early_stopping):
        return {
            'counter': early_stopping.counter,
            'best_score': early_stopping.best_score,
            'early_stop': early_stopping.early_stop,
            'val_loss_min': early_stopping.val_loss_min,
        }

    @staticmethod
    def _restore_early_stopping(early_stopping, state):
        for name in ('counter', 'best_score', 'early_stop', 'val_loss_min'):
            if name in state:
                setattr(early_stopping, name, state[name])

    def _save_training_checkpoint(
        self, checkpoint_path, completed_epochs, total_epochs,
        model_optim, scheduler, scaler, early_stopping,
    ):
        state = {
            'format_version': 1,
            'model_name': self.args.model,
            'completed_epochs': completed_epochs,
            'total_epochs': total_epochs,
            'model_state_dict': self._model_without_parallel().state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'scheduler_state_dict': (
                scheduler.state_dict() if scheduler is not None else None
            ),
            'scaler_state_dict': scaler.state_dict(),
            'early_stopping': self._early_stopping_state(early_stopping),
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if torch.cuda.is_available():
            state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()

        temporary_path = checkpoint_path + '.tmp'
        torch.save(state, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    def _resume_training(
        self, path, total_epochs, model_optim, scheduler, scaler, early_stopping,
    ):
        if not self.args.resume:
            return 0

        last_checkpoint_path = os.path.join(path, 'last_checkpoint.pth')
        best_checkpoint_path = os.path.join(path, 'checkpoint.pth')
        if not os.path.exists(last_checkpoint_path):
            if os.path.exists(best_checkpoint_path):
                self._model_without_parallel().load_state_dict(
                    torch.load(best_checkpoint_path, map_location=self.device)
                )
                print(
                    'Resume: loaded legacy best-model weights from checkpoint.pth; '
                    'optimizer and epoch state were unavailable, so the 30-epoch '
                    'schedule starts at epoch 1.'
                )
            else:
                print('Resume: no checkpoint found; starting at epoch 1.')
            return 0

        checkpoint = torch.load(
            last_checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get('model_name') != self.args.model:
            raise ValueError(
                f"Checkpoint model {checkpoint.get('model_name')!r} does not match "
                f"requested model {self.args.model!r}."
            )
        saved_total_epochs = checkpoint.get('total_epochs')
        if saved_total_epochs != total_epochs:
            raise ValueError(
                f'Checkpoint was created for {saved_total_epochs} total epochs, '
                f'but the current run requests {total_epochs}. Use a matching '
                f'configuration or start without --resume.'
            )

        self._model_without_parallel().load_state_dict(checkpoint['model_state_dict'])
        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_scheduler = checkpoint.get('scheduler_state_dict')
        if scheduler is not None and saved_scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)
        scaler_state = checkpoint.get('scaler_state_dict', {})
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        self._restore_early_stopping(
            early_stopping, checkpoint.get('early_stopping', {})
        )
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if torch.cuda.is_available() and 'cuda_rng_state_all' in checkpoint:
            # map_location=self.device also moves the serialized RNG byte
            # tensors to CUDA, but set_rng_state_all requires CPU ByteTensors.
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint['cuda_rng_state_all']]
            )
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        completed_epochs = int(checkpoint['completed_epochs'])
        print(
            f'Resume: restored last_checkpoint.pth after epoch '
            f'{completed_epochs}/{total_epochs}; continuing at epoch '
            f'{completed_epochs + 1}.'
        )
        return completed_epochs

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        if not self.args.skip_epoch_test:
            test_data, test_loader = self._get_data(flag="test")
        else:
            test_data = test_loader = None
            final_state = "disabled" if getattr(self.args, "skip_final_test", False) else "enabled"
            print(f"Per-epoch test evaluation disabled; final test is {final_state}.")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self.training_strategy.make_scheduler(
            model_optim, self.args, train_steps
        )
        use_amp = getattr(self.args, "use_amp", False) and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        total_epochs = self.args.train_epochs
        last_checkpoint_path = os.path.join(path, 'last_checkpoint.pth')
        start_epoch = self._resume_training(
            path, total_epochs, model_optim, scheduler, scaler, early_stopping
        )
        if early_stopping.early_stop and self.training_strategy.stop_on_patience:
            print('Resume: checkpoint had already reached early stopping; skipping training.')
            start_epoch = total_epochs

        for epoch in range(start_epoch, total_epochs):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    _, _, loss = self._forward_batch(
                        batch, criterion, train_data, epoch=epoch
                    )
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((total_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    self.training_strategy.clip(self.model, self.args)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    self.training_strategy.clip(self.model, self.args)
                    model_optim.step()
                self.training_strategy.batch_scheduler_step(scheduler, self.args)

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_total = torch.tensor(
                [float(np.sum(train_loss)), len(train_loss)],
                dtype=torch.float64, device=self.device,
            )
            if getattr(self.args, "use_ddp", False):
                dist.all_reduce(train_total, op=dist.ReduceOp.SUM)
            train_loss = train_total[0].item() / max(train_total[1].item(), 1)
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch=epoch)
            if test_loader is None:
                print(
                    f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                    f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}"
                )
            else:
                test_loss = self.vali(
                    test_data, test_loader, criterion, epoch=epoch
                )
                print(
                    f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                    f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} "
                    f"Test Loss: {test_loss:.7f}"
                )
            is_main = getattr(self.args, "is_main_process", True)
            if (
                epoch >= self.training_strategy.early_stopping_start_epoch
                and is_main
            ):
                early_stopping(vali_loss, self._model_without_parallel(), path)
            if getattr(self.args, "use_ddp", False):
                state = [self._early_stopping_state(early_stopping) if is_main else None]
                dist.broadcast_object_list(state, src=0)
                if not is_main:
                    self._restore_early_stopping(early_stopping, state[0])
            should_stop = (
                early_stopping.early_stop
                and self.training_strategy.stop_on_patience
            )
            if not should_stop:
                self.training_strategy.epoch_scheduler_step(
                    scheduler, model_optim, self.args, epoch + 1
                )
            if is_main:
                self._save_training_checkpoint(
                    last_checkpoint_path, completed_epochs=epoch + 1,
                    total_epochs=total_epochs, model_optim=model_optim,
                    scheduler=scheduler, scaler=scaler,
                    early_stopping=early_stopping,
                )
            if getattr(self.args, "use_ddp", False):
                dist.barrier()
            if should_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, "checkpoint.pth")
        if getattr(self.args, "is_main_process", True):
            if not os.path.exists(best_model_path):
                raise RuntimeError(
                    f"No checkpoint was produced. {self.args.model} requires more than "
                    f"{self.training_strategy.early_stopping_start_epoch} training epoch(s)."
                )
            self._model_without_parallel().load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self._model_without_parallel().load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device))

        raw_squared_sum = np.zeros(test_data.num_features, dtype=np.float64)
        raw_absolute_sum = np.zeros(test_data.num_features, dtype=np.float64)
        raw_count = np.zeros(test_data.num_features, dtype=np.float64)
        percentage_absolute_sum = np.float64(0.0)
        percentage_squared_sum = np.float64(0.0)
        percentage_count = 0
        normalized_squared_sum = np.zeros(test_data.num_features, dtype=np.float64)
        normalized_absolute_sum = np.zeros(test_data.num_features, dtype=np.float64)
        normalized_count = np.zeros(test_data.num_features, dtype=np.float64)
        test_examples = 0
        reshaped_examples = 0
        test_shape_tail = None
        dtw_list = [] if self.args.use_dtw else None
        folder_path = os.path.join(self.args.test_results, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        criterion = self._select_criterion()
        from utils.timemoe_prediction_cache import open_test_cache
        prediction_cache = open_test_cache(self.args, test_data, self._model_without_parallel())
        use_amp = getattr(self.args, "use_amp", False) and self.device.type == "cuda"
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                cached = prediction_cache.get(batch) if prediction_cache is not None else None
                if cached is None:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred_tensor, true_tensor, _ = self._forward_batch(
                            batch, criterion, test_data, epoch=None
                        )
                    if prediction_cache is not None:
                        prediction_cache.put(batch, pred_tensor.reshape(
                            len(pred_tensor), self.args.pred_len,
                            test_data.num_stations, test_data.num_features))
                else:
                    pred_tensor = torch.from_numpy(cached).flatten(start_dim=2)
                    true_tensor = batch[1][:, -self.args.pred_len:].float().flatten(start_dim=2)
                pred = pred_tensor.detach().cpu().numpy()
                true = true_tensor.detach().cpu().numpy()

                normalized_error = np.subtract(pred, true, dtype=np.float64).reshape(
                    *pred.shape[:-1], test_data.num_stations,
                    test_data.num_features,
                )
                finite = np.isfinite(normalized_error)
                normalized_error = np.where(finite, normalized_error, 0.0)
                normalized_axes = tuple(range(normalized_error.ndim - 1))
                normalized_squared_sum += np.sum(
                    normalized_error * normalized_error,
                    axis=normalized_axes,
                    dtype=np.float64,
                )
                normalized_absolute_sum += np.sum(
                    np.abs(normalized_error),
                    axis=normalized_axes,
                    dtype=np.float64,
                )
                normalized_count += finite.sum(axis=normalized_axes)

                if (
                    getattr(test_data, "scale", False)
                    and self.args.inverse
                ):
                    shape = pred.shape
                    pred = test_data.inverse_transform(
                        pred.reshape(-1, shape[-1])
                    ).reshape(shape)
                    shape = true.shape
                    true = test_data.inverse_transform(
                        true.reshape(-1, shape[-1])
                    ).reshape(shape)
                if self.args.model == "MIGN":
                    inverse = getattr(self.args, "test_inverse_transform", None)
                    if inverse is not None:
                        pred, true = inverse(pred, true)

                if pred.shape != true.shape:
                    raise ValueError(
                        f"Prediction and target shapes differ: {pred.shape} vs {true.shape}."
                    )
                if test_shape_tail is None:
                    test_shape_tail = pred.shape[1:]
                elif pred.shape[1:] != test_shape_tail:
                    raise ValueError(
                        f"Inconsistent test batch shapes: {pred.shape[1:]} vs "
                        f"{test_shape_tail}."
                    )
                test_examples += pred.shape[0]
                reshaped_examples += int(np.prod(pred.shape[:-2]))

                raw_pred = np.asarray(pred, dtype=np.float64).reshape(
                    *pred.shape[:-1], test_data.num_stations,
                    test_data.num_features,
                )
                raw_true = np.asarray(true, dtype=np.float64).reshape(
                    *true.shape[:-1], test_data.num_stations,
                    test_data.num_features,
                )
                raw_error = raw_pred - raw_true
                raw_axes = tuple(range(raw_error.ndim - 1))
                raw_squared_sum += np.sum(
                    raw_error * raw_error, axis=raw_axes, dtype=np.float64
                )
                raw_absolute_sum += np.sum(
                    np.abs(raw_error), axis=raw_axes, dtype=np.float64
                )
                raw_count += raw_error.size // test_data.num_features
                with np.errstate(divide='ignore', invalid='ignore'):
                    percentage_error = raw_error / raw_true
                percentage_absolute_sum += np.sum(
                    np.abs(percentage_error), dtype=np.float64
                )
                percentage_squared_sum += np.sum(
                    percentage_error * percentage_error, dtype=np.float64
                )
                percentage_count += percentage_error.size

                if dtw_list is not None:
                    flat_pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
                    flat_true = true.reshape(-1, true.shape[-2], true.shape[-1])
                    manhattan_distance = lambda x, y: np.abs(x - y)
                    for sample_pred, sample_true in zip(flat_pred, flat_true):
                        if len(dtw_list) % 100 == 0:
                            print("calculating dtw iter:", len(dtw_list))
                        d, _, _, _ = accelerated_dtw(
                            sample_pred.reshape(-1, 1),
                            sample_true.reshape(-1, 1),
                            dist=manhattan_distance,
                        )
                        dtw_list.append(d)
                if i % 20 == 0 and isinstance(batch, (tuple, list)) and len(batch) >= 4:
                    input_data = batch[0]
                    if (
                        torch.is_tensor(input_data)
                        and input_data.ndim == 3
                        and pred.ndim == 3
                        and true.ndim == 3
                    ):
                        input_data = input_data.detach().cpu().numpy()
                        if (
                            getattr(test_data, "scale", False)
                            and self.args.inverse
                        ):
                            shape = input_data.shape
                            input_data = test_data.inverse_transform(
                                input_data.reshape(-1, shape[-1])
                            ).reshape(shape)
                        gt = np.concatenate((input_data[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input_data[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        if prediction_cache is not None:
            prediction_cache.report()
        if test_shape_tail is None:
            raise RuntimeError("The test loader produced no batches.")
        test_shape = (test_examples, *test_shape_tail)
        reshaped_shape = (
            reshaped_examples, test_shape_tail[-2], test_shape_tail[-1]
        )
        print('test shape:', test_shape, test_shape)
        print('test shape:', reshaped_shape, reshaped_shape)

        # result save
        folder_path = os.path.join(self.args.results, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if dtw_list is not None:
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        raw_count = np.maximum(raw_count, 1.0)
        variable_mse = raw_squared_sum / raw_count
        variable_mae = raw_absolute_sum / raw_count
        mse = raw_squared_sum.sum() / raw_count.sum()
        mae = raw_absolute_sum.sum() / raw_count.sum()
        rmse = np.sqrt(mse)
        mape = percentage_absolute_sum / max(percentage_count, 1)
        mspe = percentage_squared_sum / max(percentage_count, 1)
        normalized_count = np.maximum(normalized_count, 1.0)
        normalized_variable_mse = normalized_squared_sum / normalized_count
        normalized_variable_mae = normalized_absolute_sum / normalized_count
        normalized_mse = normalized_squared_sum.sum() / normalized_count.sum()
        normalized_mae = normalized_absolute_sum.sum() / normalized_count.sum()
        variable_names = tuple(
            getattr(
                test_data,
                'variable_names',
                tuple(f'variable_{i}' for i in range(test_data.num_features)),
            )
        )
        if len(variable_names) != len(variable_mse):
            raise ValueError(
                f'Found {len(variable_names)} variable names for '
                f'{len(variable_mse)} per-variable MSE values.'
            )
        variable_mse_text = ', '.join(
            f'{name}={value:.7f}'
            for name, value in zip(variable_names, variable_mse)
        )
        variable_mae_text = ', '.join(
            f'{name}={value:.7f}'
            for name, value in zip(variable_names, variable_mae)
        )
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        print(
            'normalized_mse:{}, normalized_mae:{}'.format(
                normalized_mse, normalized_mae
            )
        )
        print('mse_by_variable: {}'.format(variable_mse_text))
        print('mae_by_variable: {}'.format(variable_mae_text))
        normalized_variable_mse_text = ', '.join(
            f'{name}={value:.7f}'
            for name, value in zip(variable_names, normalized_variable_mse)
        )
        normalized_variable_mae_text = ', '.join(
            f'{name}={value:.7f}'
            for name, value in zip(variable_names, normalized_variable_mae)
        )
        print('normalized_mse_by_variable: {}'.format(normalized_variable_mse_text))
        print('normalized_mae_by_variable: {}'.format(normalized_variable_mae_text))
        result_parent = os.path.dirname(self.args.result_file)
        if result_parent:
            os.makedirs(result_parent, exist_ok=True)
        f = open(self.args.result_file, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write(
            '\nnormalized_mse:{}, normalized_mae:{}'.format(
                normalized_mse, normalized_mae
            )
        )
        f.write('\nmse_by_variable: {}'.format(variable_mse_text))
        f.write('\nmae_by_variable: {}'.format(variable_mae_text))
        f.write('\nnormalized_mse_by_variable: {}'.format(normalized_variable_mse_text))
        f.write('\nnormalized_mae_by_variable: {}'.format(normalized_variable_mae_text))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(os.path.join(folder_path, 'metrics.npy'),
                np.array([mae, mse, rmse, mape, mspe]))
        np.savez(
            os.path.join(folder_path, 'metrics_by_variable.npz'),
            variable_names=np.asarray(variable_names),
            mse=variable_mse,
            mae=variable_mae,
            normalized_mse=np.asarray(normalized_variable_mse),
            normalized_mae=np.asarray(normalized_variable_mae),
            normalized_overall_mse=np.asarray(normalized_mse),
            normalized_overall_mae=np.asarray(normalized_mae),
        )
        # Full prediction/target arrays are intentionally not retained. Metrics
        # above are accumulated batch-wise to keep Global-scale testing bounded.

        return
