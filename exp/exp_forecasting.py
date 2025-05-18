import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import fcntl
import torch.utils
import models
from util import *
import pickle
from data.data_provider import data_provider
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .exp_basic import Exp_Basic
from tqdm import tqdm
from prettytable import PrettyTable
import logging

class Exp_Long_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        self.val_losses = []

    def _build_model(self):
        
        if self.args.model == 'ours':
            if self.args.CD:
                c_in = self.args.variable_num
            else:
                c_in = self.args.delay_emb_dim if self.args.use_time_delay else 1
            inv_mem_net = models.InvariantMemoryNet(self.args, c_in, self.args.d_model).to(self.args.device)
            model = self.model_dict[self.args.model].Model(self.args, invariant_memory_net=inv_mem_net).float().to(self.args.device)

        else:   
            model = self.model_dict[self.args.model].Model(self.args).float().to(self.args.device)
        
        return model
    
    def _get_data(self, mode, test_size=None):

        data_set, data_loader = data_provider(self.args, mode, test_size)

        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.5, patience=5, verbose=True)
        # scheduler = lr_scheduler.StepLR(model_optim, step_size=10, gamma=0.1)
        # scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.max_epoch // 10, eta_min=1e-8)
        # scheduler = lr_scheduler.StepLR(model_optim, step_size=10, gamma=0.5)
        # scheduler = None
        
        return model_optim, scheduler
    
    def _select_criterion(self):

        criterion_dict = {}
        criterion_dict['mse'] = nn.MSELoss()
        criterion_dict['sim'] = nn.CosineEmbeddingLoss()
        criterion_dict['mmd'] = MMDLoss(self.args)
        return criterion_dict
    
    def _model_run(self, input):
        
        if not self.args.shift_dataset:
            x, y = input
            x, y = x.to(self.args.device).float(), y.to(self.args.device).float()
        else:
            x = input[:, :-self.args.token_size].to(self.args.device).float() # #lookback steps
            y = input.clone().detach()[:, self.args.token_size:].to(self.args.device).float()
            
        if (self.args.temporal_memory or self.args.freq_memory) and self.args.model == 'ours':
            pred, _ = self.model(x)
        else:
            if self.args.mtp_steps > 0 and self.model.training and self.model.stage == 1:
                pred, mtp_output_list = self.model(x)
            else:
                pred = self.model(x)

        if self.args.mtp_steps > 0 and self.model.training and self.model.stage == 1:
            return pred, mtp_output_list, y
        else:
            return pred, y
    
    def _model_autoregression(self, x, y, pred_steps, test_flag=False):
        x, y = x.to(self.args.device).float(), y.to(self.args.device).float()
        if test_flag:
            if self.args.data_type == 'sim':
                x = x[:, -self.args.test_warmup_steps:]
            elif self.args.data_type == 'real':
                x = x[:, -self.args.lookback:]
            else:
                raise ValueError('data_type must be in [sim, real]')
        else:
            x = x[:, -self.args.lookback:]

        y_hat = self.model.autoregression(x, future=pred_steps, token_size=self.args.token_size, long_term=True)

        return y_hat
    
    def _calc_loss(self, pred, label, mtp_output_list=None, input=None, mean=None, variance=None):

        loss = self.criterion['mse'](pred, label)
        
        if self.args.model == 'ours':
            if (self.args.temporal_memory or self.args.freq_memory) and self.args.use_inv_loss and self.args.shift_dataset:

                label_input, _, __ = self.model._normalize_self(label, reshape_to_ci=True if not self.args.CD else False)
                future_prompt_dict = self.model.invariant_memory_net(label_input, norm=True)
                future_time_prompt, future_freq_prompt = future_prompt_dict['time_prompt'], future_prompt_dict['freq_prompt']
                # future_prompt_input = model.backbone.dilatedconv(label_input)
                # future_prompt = model.backbone.temporal_memory(future_prompt_input)['out']

                pred_input, _, __ = self.model._normalize_self(pred, reshape_to_ci=True if not self.args.CD else False)
                pred_prompt_dict = self.model.invariant_memory_net(pred_input, norm=True)
                pred_time_prompt, pred_freq_prompt = pred_prompt_dict['time_prompt'], pred_prompt_dict['freq_prompt']
                # pred_prompt_input = model.backbone.dilatedconv(pred_input)
                # pred_prompt = model.backbone.temporal_memory(pred_prompt_input)['out']

                if self.args.temporal_memory:
                    btimen, T, F = pred_time_prompt.shape
                else:
                    btimen, T, F = pred_freq_prompt.shape

                if self.args.temporal_memory:
                    if self.args.invariant_supervise_type == 'time-wise':
                        pred_time_prompt = pred_time_prompt.reshape(btimen*T, F)
                        future_time_prompt = future_time_prompt.reshape(btimen*T, F)
                    elif self.args.invariant_supervise_type == 'mean':
                        pred_time_prompt = pred_time_prompt.mean(dim=1) #(B*N, F)
                        future_time_prompt = future_time_prompt.mean(dim=1) # (B*N, F)
                    else:
                        raise NotImplementedError
                    
                    loss += self.args.lambda_time_rec * self.criterion['sim'](pred_time_prompt, future_time_prompt, target=torch.ones(pred_time_prompt.shape[0], device=self.args.device))

                if self.args.freq_memory:
                    if self.args.invariant_supervise_type == 'time-wise':
                        pred_freq_prompt = pred_freq_prompt.reshape(btimen*T, F)
                        future_freq_prompt = future_freq_prompt.reshape(btimen*T, F)
                    elif self.args.invariant_supervise_type == 'mean':
                        pred_freq_prompt = pred_freq_prompt.mean(dim=1) #(B*N, F)
                        future_freq_prompt = future_freq_prompt.mean(dim=1) # (B*N, F)
                    else:
                        raise NotImplementedError
                    
                    loss += self.args.lambda_freq_rec * self.criterion['sim'](pred_freq_prompt, future_freq_prompt, target=torch.ones(pred_freq_prompt.shape[0], device=self.args.device))
        
        if mtp_output_list is not None:

            mtp_losses = 0

            for k, mtp_output in enumerate(mtp_output_list):
                mtp_target = label[:, (k+1)*self.args.token_size:, :]
                mtp_losses += self.criterion['mse'](mtp_output, mtp_target)
            
            mtp_losses /= len(mtp_output_list)

            mtp_losses = mtp_losses * self.args.lambda_mtp_loss

            loss = loss + mtp_losses
        
        if (self.args.lambda_uncond_mmd > 0 or self.args.lambda_cond_mmd > 0) and (self.model.stage == 2):

            mmd_loss = self.criterion['mmd'](input, pred, label, mean, variance)
            
            loss = loss + mmd_loss
            


        return loss

    def validation(self, val_loader, criterion, logger, info_dict=None, is_print=False, stage=None):

        pred_long, true_long = [], []
        self.model.eval()
        self.model.stage = stage
        with torch.no_grad():
            pred_loss = 0.
            if is_print:
                val_loader = tqdm(val_loader)
            
            for i, input in enumerate(val_loader):
                
                if self.args.val_long or stage == 2:
                    x, y = input
                    pred = self._model_autoregression(x, y, pred_steps=y.shape[1])
                    loss = self._calc_loss(pred, y, input=x, mean=info_dict['train_mean'], variance=info_dict['train_var'])
                else:
                    pred, y = self._model_run(input)
                    loss = self._calc_loss(pred, y)
                # loss = criterion['mse'](pred, y)
                
                    
                pred_loss += loss.item()
                pred_long.append(pred.cpu().numpy())
                true_long.append(y.cpu().numpy())
            
            pred_loss /= len(val_loader)
            if info_dict["epoch"] > 0:
                self.val_losses.append(pred_loss)
            pred_long = np.concatenate(pred_long, axis=0)
            true_long = np.concatenate(true_long, axis=0)

            stage_str = "Stage 1" if stage == 1 else "Stage 2"
            if is_print:
                val_loader.set_description(f"{stage_str} Epoch {info_dict['epoch']}/{self.args.max_epoch} | val-loss={pred_loss:.5f} | train-loss={info_dict['average_train_loss']:.5f}")
                print(f"{stage_str} Epoch {info_dict['epoch']}/{self.args.max_epoch} | val-loss={pred_loss:.5f} | train-loss={info_dict['average_train_loss']:.5f}")
                logger.info(f"{stage_str} Epoch {info_dict['epoch']}/{self.args.max_epoch} | val-loss={pred_loss:.5f} | train-loss={info_dict['average_train_loss']:.5f}")

            if info_dict['scheduler'] is not None:
                info_dict['scheduler'].step(pred_loss)

        if self.args.val_long or stage == 2:
            smape_dict, vpt, cd_rmse, dh, kld = test_metrics_long(true_long, pred_long, self.args)
            mae, mse, r2 = test_metrics_short(true_long, pred_long, self.args.short_pred_length)

            if is_print:
                print(f"Validation @ {stage_str} Epoch[{info_dict['epoch']}/{self.args.max_epoch}]")
                print('----------------------------------------------------------------')
                print(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
                print('----------------------------------------------------------------')
                pred_T = true_long.shape[1]
                print(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
                print(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

                logger.info(f"Validation @ {stage_str} Epoch[{info_dict['epoch']}/{self.args.max_epoch}]")
                logger.info('----------------------------------------------------------------')
                logger.info(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
                logger.info('----------------------------------------------------------------')
                logger.info(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
                logger.info(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

        self.model.train()

        return {'val_loss': pred_loss}
        # scheduler need to update
        
    def train(self, is_print=False, random_seed=729):

        log_dir = os.path.join(self.args.log_dir, f'seed{random_seed}')
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        # 获取训练专用的日志器
        train_logger = logging.getLogger("train")
        train_logger.setLevel(logging.INFO)
        train_logger.propagate = False  # 防止日志传播到根日志器
        
        # 移除已有处理器，避免重复
        for handler in train_logger.handlers[:]:
            train_logger.removeHandler(handler)
        
        # 添加文件处理器，记录INFO及以上日志
        file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        
        # 添加控制台处理器，仅显示WARNING及以上日志
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        
        train_logger.addHandler(file_handler)
        train_logger.addHandler(console)


        # 假设这是一个类的内部代码，self.model 已定义
        trainable_params = 0
        non_trainable_params = 0
        trainable_size = 0
        non_trainable_size = 0
        total_params = sum(p.numel() for p in self.model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in self.model.parameters())

        for param in self.model.parameters():
            num_elements = param.numel()
            size_bytes = num_elements * param.element_size()
            if param.requires_grad:
                trainable_params += num_elements
                trainable_size += size_bytes
            else:
                non_trainable_params += num_elements
                non_trainable_size += size_bytes

        # 转换为 K（千）单位，保留两位小数
        trainable_params_k = trainable_params / (1024 * 1024)
        non_trainable_params_k = non_trainable_params / (1024 * 1024)
        total_params_k = total_params / (1024 * 1024)  # 直接用 total_params 转换

        # 转换为 MB，保留两位小数
        trainable_size_mb = trainable_size / (1024 * 1024)
        non_trainable_size_mb = non_trainable_size / (1024 * 1024)
        total_size_mb = total_size / (1024 * 1024)

        # 创建 PrettyTable 并设置列名
        param_table = PrettyTable()
        param_table.field_names = ["Trainable", "Non-Trainable", "Total", "Model Size"]

        # 添加一行，显示参数量 (K) 和内存大小 (MB)
        param_table.add_row([
            f"{trainable_params_k:7f} M",
            f"{non_trainable_params_k:.7f} M",
            f"{total_params_k:.7f} M",
            f"{total_size_mb:.7f} MB"
        ])

        # 打印表格
        print(param_table)
        train_logger.info(param_table)

        best_val_loss = float('inf')
        early_stop_patience = self.args.early_stop_patience
        patience_counter = 0

        train_dataset, train_loader = self._get_data('train_stage1')
        val_dataset, val_loader = self._get_data('val_stage1')

        if not self.args.skip_stage2:
            train_dataset_stage2, train_loader_stage2 = self._get_data('train_stage2')
            train_mean, train_var = torch.from_numpy(train_dataset_stage2.channel_mean).to(self.args.device), torch.from_numpy(train_dataset_stage2.channel_variance).to(self.args.device)
            
            val_dataset_stage2, val_loader_stage2 = self._get_data('val_stage2')


        model_optim, model_scheduler = self._select_optimizer()
        criterion = self._select_criterion()
        self.criterion = criterion

        # self.validation(val_loader, criterion, info_dict={'epoch': 0, 'average_train_loss': 0, 'scheduler': model_scheduler}, is_print=is_print)

        if self.args.train_from_ckpts:
            ckpt_path = os.path.join(log_dir, "checkpoints", f"epoch-{self.args.ckpts_epoch}.pth")
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt)
            print(f'Load model from {ckpt_path}')
            train_logger.info(f'Load model from {ckpt_path}')
        
        start_epoch = 1 if not self.args.train_from_ckpts else self.args.ckpts_epoch + 1

        stage1_epochs = self.args.stage1_max_epoch
        stage2_linear_epochs = self.args.stage2_linear_epoch
        stage2_full_epochs = self.args.max_epoch - stage2_linear_epochs - stage1_epochs

        if not self.args.skip_stage1:
            ###### Stage 1 training epochs.....
            self.model.stage = 1
            for epoch in range(start_epoch, stage1_epochs + 1):
                train_losses = []
                self.model.train()
                if is_print:
                    train_loader = tqdm(train_loader)
                
                model_optim.zero_grad()

                for i, input in enumerate(train_loader):
                    
                    if self.args.mtp_steps > 0:
                        pred, mtp_output_list, label = self._model_run(input)
                        loss = self._calc_loss(pred, label, mtp_output_list=mtp_output_list)
                        
                    else:
                        pred, label = self._model_run(input)
                        # loss = criterion['mse'](pred, label)
                        loss = self._calc_loss(pred, label)
                        
                    train_losses.append(loss.item())
                    loss.backward()
                    model_optim.step()
                    model_optim.zero_grad()

                    if is_print:
                        train_loader.set_description(f'Stage 1 Epoch [{epoch}/{stage1_epochs}] | train-loss={np.mean(train_losses):.5f}')

                average_train_loss = np.mean(train_losses)
                if is_print:
                    print(f'Stage 1 Epoch [{epoch}/{stage1_epochs}] | train-loss={average_train_loss:.5f}')
                    train_logger.info(f'Stage 1 Epoch [{epoch}/{stage1_epochs}] | train-loss={average_train_loss:.5f}')

                if epoch % self.args.val_interval == 0 or epoch in [1, stage1_epochs]:
                    if not self.args.skip_stage2:
                        val_info = self.validation(val_loader, criterion, logger=train_logger, info_dict={'epoch': epoch, 'average_train_loss': average_train_loss, 'scheduler': model_scheduler, 'train_mean': train_mean, 'train_var': train_var}, is_print=is_print, stage=1)            
                    else:
                        val_info = self.validation(val_loader, criterion, logger=train_logger, info_dict={'epoch': epoch, 'average_train_loss': average_train_loss, 'scheduler': model_scheduler}, is_print=is_print, stage=1)
                    current_val_loss = val_info['val_loss']
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', 'best_model_stage1.pth'))
                        print(f'Saved best model (Stage 1) at epoch {epoch} with val_loss={best_val_loss:.5f}')
                        train_logger.info(f'Saved best model (Stage 1) at epoch {epoch} with val_loss={best_val_loss:.5f}')
                    else:
                        patience_counter += 1
                        train_logger.info(f'Early stopping counter: {patience_counter}/{early_stop_patience}')
                        if patience_counter >= early_stop_patience:
                            print(f'Early stopping triggered at epoch {epoch} (Stage 1)')
                            train_logger.info(f'Early stopping triggered at epoch {epoch} (Stage 1)')
                            break
                #########
                # if model_scheduler is not None:
                #     model_scheduler.step()
                #########

                if epoch % self.args.save_interval == 0:
                    torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', f'stage1_epoch-{epoch}.pth'))
            
            torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', 'stage1_final.pth'))
            print("Stage 1 completed. Starting Stage 2: Multi-step Autoregressive Prediction")
            train_logger.info("Stage 1 completed. Starting Stage 2: Multi-step Autoregressive Prediction")

        if not self.args.skip_stage2:

            try:
                ckpt_path = os.path.join(log_dir, 'checkpoints', 'best_model_stage1.pth')
                ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(ckpt)
                print(f'Loaded best model from Stage 1: {ckpt_path}')
                train_logger.info(f'Loaded best model from Stage 1: {ckpt_path}')
            except:
                print(f'Failed to load best model from Stage 1: {ckpt_path}')
                train_logger.info(f'Failed to load best model from Stage 1: {ckpt_path}')
            

            # Set new optimizer for stage 2
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.finetune_lr)  # 使用新的学习率
            model_scheduler = lr_scheduler.ReduceLROnPlateau(model_optim, mode='min', factor=0.5, patience=5, verbose=True)
            patience_counter = 0
            best_val_loss = float('inf')

            print("Stage 2 - Multi-step Autoregressive Prediction")
            train_logger.info("Stage 2 - Linear Probing: Freezing backbone, training output layer")
            for name, param in self.model.named_parameters():
                if 'head' not in name:  
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            self.model.stage = 2
            
            for epoch in range(stage1_epochs, stage1_epochs + stage2_linear_epochs + 1):

                train_losses = []
                self.model.train()
                if is_print:
                    train_loader = tqdm(train_loader_stage2)
                else:
                    train_loader = train_loader_stage2
                
                for i, input in enumerate(train_loader):
                    x, y = input
                    pred_steps = y.shape[1]
                    x, y = x.to(self.args.device).float(), y.to(self.args.device).float()
                    model_optim.zero_grad()
                    pred = self._model_autoregression(x, y, pred_steps=pred_steps) ### !!!!!!!!
                    loss = self._calc_loss(pred, y, input=x, mean=train_mean, variance=train_var)
                    ####### debug code #######
                    # os.makedirs(os.path.join(log_dir, 'debug'), exist_ok=True)
                    # torch.save(pred, os.path.join(log_dir, 'debug', f'pred.pt'))
                    # torch.save(y, os.path.join(log_dir, 'debug', f'y.pt'))
                    # torch.save(x, os.path.join(log_dir, 'debug', f'x.pt'))
                    # torch.save(train_mean, os.path.join(log_dir, 'debug', f'train_mean.pt'))
                    # torch.save(train_var, os.path.join(log_dir, 'debug', f'train_var.pt'))
                    # assert False
                    ########################
                    train_losses.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()
                    if is_print:
                        train_loader.set_description(f'Stage 2 Linear Epoch [{epoch}/{stage1_epochs + stage2_linear_epochs}] | train-loss={np.mean(train_losses):.5f}')

                
                average_train_loss = np.mean(train_losses)
                if is_print:
                    print(f'Stage 2 Linear Epoch [{epoch}/{stage1_epochs + stage2_linear_epochs}] | train-loss={average_train_loss:.5f}')
                    train_logger.info(f'Stage 2 Linear Epoch [{epoch}/{stage1_epochs + stage2_linear_epochs}] | train-loss={average_train_loss:.5f}')
                
                if epoch % self.args.val_interval == 0 or epoch in [stage1_epochs + 1, stage1_epochs + stage2_linear_epochs]:
                    val_info = self.validation(val_loader_stage2, criterion, logger=train_logger, 
                                            info_dict={'epoch': epoch, 'average_train_loss': average_train_loss, 'scheduler': model_scheduler, 'train_mean': train_mean, 'train_var': train_var}, 
                                            is_print=is_print, stage=2)
                    current_val_loss = val_info['val_loss']
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', 'best_model_stage2_linear.pth'))
                        print(f'Saved best model (Stage 2 Linear) at epoch {epoch} with val_loss={best_val_loss:.5f}')
                        train_logger.info(f'Saved best model (Stage 2 Linear) at epoch {epoch} with val_loss={best_val_loss:.5f}')
                    else:
                        patience_counter += 1
                        train_logger.info(f'Early stopping counter: {patience_counter}/{early_stop_patience}')
                        if patience_counter >= early_stop_patience:
                            print(f'Early stopping triggered at epoch {epoch} (Stage 2 Linear)')
                            train_logger.info(f'Early stopping triggered at epoch {epoch} (Stage 2 Linear)')
                            break
            
            # 保存 Linear Probing 阶段的最终模型
            torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', 'stage2_linear_final.pth'))
            print("Stage 2 Linear Probing completed")
            train_logger.info("Stage 2 Linear Probing completed")

            # 第二阶段 - Fully Fine-tuning
            
            patience_counter = 0
            best_val_loss = float('inf')
            print("Stage 2 - Fully Fine-tuning: Training all layers")
            train_logger.info("Stage 2 - Fully Fine-tuning: Training all layers")
            # 解冻所有参数
            for param in self.model.parameters():
                param.requires_grad = True
            
            try:
                # 从 Linear Probing 的最佳模型加载
                ckpt_path = os.path.join(log_dir, 'checkpoints', 'best_model_stage2_linear.pth')
                ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(ckpt)
                print(f'Loaded best model from Stage 2 Linear: {ckpt_path}')
                train_logger.info(f'Loaded best model from Stage 2 Linear: {ckpt_path}')
            except:
                print(f'Failed to load best model from Stage 2 Linear: {ckpt_path}')
                train_logger.info(f'Failed to load best model from Stage 2 Linear: {ckpt_path}')
            
            self.model.stage = 2
            for epoch in range(stage1_epochs + stage2_linear_epochs + 1, stage1_epochs + stage2_linear_epochs + stage2_full_epochs + 1):
                train_losses = []
                self.model.train()
                if is_print:
                    train_loader = tqdm(train_loader_stage2)
                else:
                    train_loader = train_loader_stage2
                
                for i, input in enumerate(train_loader):
                    x, y = input
                    pred_steps = y.shape[1]
                    model_optim.zero_grad()
                    pred = self._model_autoregression(x, y, pred_steps=pred_steps) ### !!!!!!!!
                    
                    loss = self._calc_loss(pred, y, input=x, mean=train_mean, variance=train_var)
                    train_losses.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()
                    if is_print:
                        train_loader.set_description(f'Stage 2 Full Epoch [{epoch}/{stage1_epochs + stage2_linear_epochs + stage2_full_epochs}] | train-loss={np.mean(train_losses):.5f}')
                
                average_train_loss = np.mean(train_losses)
                if is_print:
                    print(f'Stage 2 Full Epoch [{epoch}/{stage1_epochs + stage2_linear_epochs + stage2_full_epochs}] | train-loss={average_train_loss:.5f}')
                    train_logger.info(f'Stage 2 Full Epoch [{epoch}/{stage1_epochs + stage2_linear_epochs + stage2_full_epochs}] | train-loss={average_train_loss:.5f}')
                
                if epoch % self.args.val_interval == 0 or epoch in [stage1_epochs + stage2_linear_epochs + 1, stage1_epochs + stage2_linear_epochs + stage2_full_epochs]:
                    val_info = self.validation(val_loader_stage2, criterion, logger=train_logger, 
                                            info_dict={'epoch': epoch, 'average_train_loss': average_train_loss, 'scheduler': model_scheduler, 'train_mean': train_mean, 'train_var': train_var}, 
                                            is_print=is_print, stage=2)
                    current_val_loss = val_info['val_loss']
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', 'best_model_stage2_full.pth'))
                        print(f'Saved best model (Stage 2 Full) at epoch {epoch} with val_loss={best_val_loss:.5f}')
                        train_logger.info(f'Saved best model (Stage 2 Full) at epoch {epoch} with val_loss={best_val_loss:.5f}')
                    else:
                        patience_counter += 1
                        train_logger.info(f'Early stopping counter: {patience_counter}/{early_stop_patience}')
                        if patience_counter >= early_stop_patience:
                            print(f'Early stopping triggered at epoch {epoch} (Stage 2 Full)')
                            train_logger.info(f'Early stopping triggered at epoch {epoch} (Stage 2 Full)')
                            break
                
                if epoch % self.args.save_interval == 0:
                    torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', f'stage2_full_epoch-{epoch}.pth'))
        
        torch.save(self.model.state_dict(), os.path.join(log_dir, 'checkpoints', f'epoch-{self.args.max_epoch}.pth'))
        # plot_start_epoch = 0 if not self.args.train_from_ckpts else self.args.ckpts_epoch
        plt.figure(figsize=(6, 4))
        plt.plot(self.val_losses)
        plt.legend(['pred_loss'])
        plt.xlabel('epoch')
        plt.xticks(ticks=range(len(self.val_losses)))
        # plt.xticks(ticks=range(len(self.val_losses)), labels=range(plot_start_epoch, self.args.max_epoch+1, self.args.val_interval))
        plt.tight_layout(); 
        if not self.args.train_from_ckpts:
            plt.savefig(log_dir+'/val_loss_curve.jpg', dpi=300)
        else:
            plt.savefig(log_dir+'/val_loss_curve_retrain.jpg', dpi=300)

    def test(self, is_print=False, ood_test=False, finetune_epochs=0, finetune_fracs=0.0, random_seed=729):
        # prepare
        log_dir = os.path.join(self.args.log_dir, f'seed{random_seed}')
        os.makedirs(os.path.join(log_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}'), exist_ok=True)

        # 获取测试专用的日志器
        test_logger = logging.getLogger("test")
        test_logger.setLevel(logging.INFO)
        test_logger.propagate = False  # 防止日志传播到根日志器
        
        # 移除已有处理器，避免重复
        for handler in test_logger.handlers[:]:
            test_logger.removeHandler(handler)
        
        # 添加文件处理器，记录INFO及以上日志
        if not self.args.get_test_time_only:
            file_handler = logging.FileHandler(os.path.join(log_dir, "test.log"), mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        else:
            file_handler = logging.FileHandler(os.path.join(log_dir, "test_time.log"), mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
            
        # 添加控制台处理器，仅显示WARNING及以上日志
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
        
        test_logger.addHandler(file_handler)
        test_logger.addHandler(console)

        if ood_test:
            os.makedirs(os.path.join(self.args.finetune_log_dir, 'test'), exist_ok=True)

        test_dataset_longt, test_loader_longt = self._get_data('test')

        with torch.no_grad():
            if ood_test and finetune_epochs > 0:
                ckpt_path = os.path.join(self.args.finetune_log_dir, f'seed{random_seed}', "checkpoints", f"finetune-epoch-{self.args.finetune_epochs}.pth")
                ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(ckpt)
                print(f'Load model from {ckpt_path}')
                test_logger.info(f'Load model from {ckpt_path}')
            
            elif ood_test and (finetune_epochs == 0):
                if not self.args.from_scratch:
                    ckpt_path = os.path.join(log_dir, "checkpoints", f"epoch-{self.args.max_epoch}.pth")
                    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                    self.model.load_state_dict(ckpt)
                    print(f'Load model from {ckpt_path}')
                    test_logger.info(f'Load model from {ckpt_path}')
                else:
                    print('From Scratch Model with Zero-shot performance......')
                    test_logger.info('From Scratch Model with Zero-shot performance......')
            else:
                try:
                    if self.args.test_single_step_only:
                        ckpt_path = os.path.join(log_dir, 'checkpoints', f'best_model_stage1.pth')
                    else:
                        if self.args.load_best:
                            if self.args.skip_stage2:
                                ckpt_path = os.path.join(log_dir, 'checkpoints', 'best_model_stage1.pth')
                            else:
                                ckpt_path = os.path.join(log_dir, 'checkpoints', 'best_model_stage2_full.pth')                
                        else: # load final
                            ckpt_path = os.path.join(log_dir, "checkpoints", f"epoch-{self.args.max_epoch}.pth")
                    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                    self.model.load_state_dict(ckpt)
                    print(f'Load model from {ckpt_path}')
                    test_logger.info(f'Load model from {ckpt_path}')
                except:
                    print(f'Failed to load model from {ckpt_path}')
                    test_logger.info(f'Failed to load model from {ckpt_path}')
                    print(f'Use random initialized model for testing......')
                    test_logger.info(f'Use random initialized model for testing......')

            pred_long, true_long = [], []
            self.model.eval()

            if self.args.get_test_time_only:
                get_model_size(self.model, test_logger)
                inft, std, fps = get_inference_times(self.args, self.model)
                print(f'Inference time: {inft:.5f}s, std: {std:.5f}s, FPS: {fps:.5f}')
                test_logger.info(f'Inference time: {inft:.5f}s, std: {std:.5f}s, FPS: {fps:.5f}')
                return
            
            if is_print:
                test_loader_longt = tqdm(test_loader_longt, total=len(test_loader_longt))
            
            for idx, (x, y) in enumerate(test_loader_longt):
                pred = self._model_autoregression(x, y, pred_steps=y.shape[1], test_flag=True)
                pred_long.append(pred.cpu().numpy())
                true_long.append(y.cpu().numpy())
            
            pred_long = np.concatenate(pred_long, axis=0)
            true_long = np.concatenate(true_long, axis=0)

            if self.args.save_pred:
                if ood_test:
                    if self.args.test_single_step_only:
                        pred_long_filename = f'ood_pred_long_epoch_zero_shot_stage1.npy' if finetune_epochs == 0 else f'ood_pred_long_epoch_{finetune_epochs}_frac_{finetune_fracs}_stage1.npy'
                        true_long_filename = f'ood_true_long_epoch_zero_shot_stage1.npy' if finetune_epochs == 0 else f'ood_true_long_epoch_{finetune_epochs}_frac_{finetune_fracs}_stage1.npy'
                        np.save(os.path.join(self.args.finetune_log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', pred_long_filename), pred_long)
                        np.save(os.path.join(self.args.finetune_log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', true_long_filename), true_long)
                    
                    else:
                        pred_long_filename = f'ood_pred_long_epoch_zero_shot.npy' if finetune_epochs == 0 else f'ood_pred_long_epoch_{finetune_epochs}_frac_{finetune_fracs}.npy'
                        true_long_filename = f'ood_true_long_epoch_zero_shot.npy' if finetune_epochs == 0 else f'ood_true_long_epoch_{finetune_epochs}_frac_{finetune_fracs}.npy'
                        
                        if self.args.load_best:
                            np.save(os.path.join(self.args.finetune_log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_{pred_long_filename}'), pred_long)
                            np.save(os.path.join(self.args.finetune_log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_{true_long_filename}'), true_long)
                        else:
                            np.save(os.path.join(self.args.finetune_log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', pred_long_filename), pred_long)
                            np.save(os.path.join(self.args.finetune_log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', true_long_filename), true_long)
                else:

                    if self.args.test_single_step_only:
                        np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'pred_long_epoch_{self.args.max_epoch}_stage1.npy'), pred_long)
                        np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'true_long_epoch_{self.args.max_epoch}_stage1.npy'), true_long)
                    else:
                        if self.args.load_best:
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_pred_long_epoch_{self.args.max_epoch}.npy'), pred_long)
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_true_long_epoch_{self.args.max_epoch}.npy'), true_long)
                        
                        else:
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'pred_long_epoch_{self.args.max_epoch}.npy'), pred_long)
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'true_long_epoch_{self.args.max_epoch}.npy'), true_long)
        
        # if (self.args.data_type != 'real') and (self.args.data_type != 'EEG') and (self.args.data_type != 'ECG'):
        smape_dict, vpt, cd_rmse, dh, kld = test_metrics_long(true_long, pred_long, self.args)
        mae, mse, r2 = test_metrics_short(true_long, pred_long, self.args.short_pred_length)

        if is_print:
            print('----------------------------------------------------------------')
            print(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
            print('----------------------------------------------------------------')
            pred_T = true_long.shape[1]
            print(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.5: {smape_dict[int(pred_T * 0.5)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
            print(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

            test_logger.info('----------------------------------------------------------------')
            test_logger.info(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
            test_logger.info('----------------------------------------------------------------')
            test_logger.info(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.5: {smape_dict[int(pred_T * 0.5)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
            test_logger.info(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

        if ood_test:
            filename = f'{self.args.finetune_log_dir}/w{self.args.lookback}_t{self.args.data_forecast_length}_twarm_{self.args.test_warmup_steps}'
            if self.args.test_single_step_only:
                filename += '_stage1'
            with open(filename + '.txt','a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines(f'{random_seed}, {vpt:.5f}, {cd_rmse:.5f}, {dh:.5f}, {kld:.5f}, {mae:.5f}, {mse:.5f}, {r2:.5f}, {smape_dict[int(pred_T * 0.1)]:.5f},{smape_dict[int(pred_T * 0.2)]:.5f}, {smape_dict[int(pred_T * 0.4)]:.5f}, {smape_dict[int(pred_T * 0.5)]:.5f}, {smape_dict[int(pred_T * 0.6)]:.5f}, {smape_dict[int(pred_T * 0.8)]:.5f}, {smape_dict[int(pred_T)]:.5f}\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)

        else:
            filename = f'{self.args.log_dir}/w{self.args.lookback}_t{self.args.data_forecast_length}_twarm_{self.args.test_warmup_steps}'
            if self.args.test_single_step_only:
                filename += '_stage1'

            with open(filename + '.txt','a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines(f'{random_seed}, {vpt:.5f}, {cd_rmse:.5f}, {dh:.5f}, {kld:.5f}, {mae:.5f}, {mse:.5f}, {r2:.5f}, {smape_dict[int(pred_T * 0.1)]:.5f},{smape_dict[int(pred_T * 0.2)]:.5f}, {smape_dict[int(pred_T * 0.4)]:.5f}, {smape_dict[int(pred_T * 0.5)]:.5f}, {smape_dict[int(pred_T * 0.6)]:.5f}, {smape_dict[int(pred_T * 0.8)]:.5f}, {smape_dict[int(pred_T)]:.5f}\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)


        if self.args.multi_init_test:
            
            if self.args.exp_name == 'standard':
                if self.args.system == 'ECG':
                    test_size_list = [20]
                elif self.args.system == 'EEG':
                    if self.args.long_test_length == 600:
                        test_size_list = [13]
                    else:
                        assert self.args.long_test_length == 300
                        test_size_list = [24]
                else:
                    test_size_list = [10, 30, 50, 70, 100]
            elif self.args.exp_name == 'mi':
                raise NotImplementedError
            elif self.args.exp_name == 'mp_full':
                test_size_list = [90, 270, 450, 630, 900]
            elif self.args.exp_name == 'mp_in' or self.args.exp_name == 'mp_ex':
                test_size_list = [30, 90, 150, 210, 300]
            else:
                raise NotImplementedError

            # if self.args.exp_name != 'standard':
            #     test_folder_name = f'{self.args.exp_name}_' + 'test'

            for idx, test_size in enumerate(test_size_list):

                test_dataset_large, test_loader_large = self._get_data('test_large', test_size)

                with torch.no_grad():
            
                    pred_long, true_long = [], []
                    self.model.eval()

                    if is_print:
                        test_loader_large = tqdm(test_loader_large, total=len(test_loader_large))
                    
                    for idx, (x, y) in enumerate(test_loader_large):
                        pred = self._model_autoregression(x, y, pred_steps=y.shape[1], test_flag=True)

                        pred_long.append(pred.cpu().numpy())
                        true_long.append(y.cpu().numpy())
                    
                    pred_long = np.concatenate(pred_long, axis=0)
                    true_long = np.concatenate(true_long, axis=0)

                    if self.args.save_pred:

                        if self.args.test_single_step_only:
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'pred_long_epoch_{self.args.max_epoch}_test_size_{test_size}_stage1.npy'), pred_long)
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'true_long_epoch_{self.args.max_epoch}_test_size_{test_size}_stage1.npy'), true_long)

                        else:
                            if self.args.load_best:
                                np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_pred_long_epoch_{self.args.max_epoch}_test_size_{test_size}.npy'), pred_long)
                                np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_true_long_epoch_{self.args.max_epoch}_test_size_{test_size}.npy'), true_long)
                            else:
                                np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'pred_long_epoch_{self.args.max_epoch}_test_size_{test_size}.npy'), pred_long)
                                np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'true_long_epoch_{self.args.max_epoch}_test_size_{test_size}.npy'), true_long)
                    
                smape_dict, vpt, cd_rmse, dh, kld = test_metrics_long(true_long, pred_long, self.args)
                mae, mse, r2 = test_metrics_short(true_long, pred_long, self.args.short_pred_length)

                if is_print:
                    print('Metrics for test size:', test_size)
                    print('----------------------------------------------------------------')
                    print(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
                    print('----------------------------------------------------------------')
                    pred_T = true_long.shape[1]
                    print(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.5: {smape_dict[int(pred_T * 0.5)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
                    print(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

                    test_logger.info('Metrics for test size: %d', test_size)
                    test_logger.info('----------------------------------------------------------------')
                    test_logger.info(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
                    test_logger.info('----------------------------------------------------------------')
                    test_logger.info(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.5: {smape_dict[int(pred_T * 0.5)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
                    test_logger.info(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

                filename = f'{self.args.log_dir}/w{self.args.lookback}_t{self.args.data_forecast_length}_test_size_{test_size}_twarm_{self.args.test_warmup_steps}'
                if self.args.test_single_step_only:
                    filename += '_stage1'
                with open(filename + '.txt','a') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.writelines(f'{random_seed}, {vpt:.5f}, {cd_rmse:.5f}, {dh:.5f}, {kld:.5f}, {mae:.5f}, {mse:.5f}, {r2:.5f}, {smape_dict[int(pred_T * 0.1)]:.5f},{smape_dict[int(pred_T * 0.2)]:.5f}, {smape_dict[int(pred_T * 0.4)]:.5f}, {smape_dict[int(pred_T * 0.5)]:.5f}, {smape_dict[int(pred_T * 0.6)]:.5f}, {smape_dict[int(pred_T * 0.8)]:.5f}, {smape_dict[int(pred_T)]:.5f}\n')
                    f.flush()
                    fcntl.flock(f, fcntl.LOCK_UN)
            
                if self.args.exp_type != 'others' and idx == len(test_size_list) - 1:
                    print_test_metrics(self.args, smape_dict, vpt, cd_rmse, mae, mse, dh, kld)

        if self.args.test_ood and self.args.exp_name == 'mi':

            test_dataset_ood, test_loader_ood = self._get_data('test_ood')
            with torch.no_grad():
        
                pred_long, true_long = [], []
                self.model.eval()

                if is_print:
                    test_loader_ood = tqdm(test_loader_ood, total=len(test_loader_ood))
                
                for idx, (x, y) in enumerate(test_loader_ood):
                    pred = self._model_autoregression(x, y, pred_steps=y.shape[1], test_flag=True)

                    pred_long.append(pred.cpu().numpy())
                    true_long.append(y.cpu().numpy())
                
                pred_long = np.concatenate(pred_long, axis=0)
                true_long = np.concatenate(true_long, axis=0)

                if self.args.save_pred:

                    if self.args.test_single_step_only:
                        np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'pred_long_epoch_{self.args.max_epoch}_ood_{self.args.ood_test_traj_num}_stage1.npy'), pred_long)
                        np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'true_long_epoch_{self.args.max_epoch}_ood_{self.args.ood_test_traj_num}_stage1.npy'), true_long)
                    else:
                        if self.args.load_best:
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_pred_long_epoch_{self.args.max_epoch}_ood_{self.args.ood_test_traj_num}.npy'), pred_long)
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'best_true_long_epoch_{self.args.max_epoch}_ood_{self.args.ood_test_traj_num}.npy'), true_long)
                        else:
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'pred_long_epoch_{self.args.max_epoch}_ood_{self.args.ood_test_traj_num}.npy'), pred_long)
                            np.save(os.path.join(log_dir, 'test', f'twarm_{self.args.test_warmup_steps}', f'true_long_epoch_{self.args.max_epoch}_ood_{self.args.ood_test_traj_num}.npy'), true_long)
            
            smape_dict, vpt, cd_rmse, dh, kld = test_metrics_long(true_long, pred_long, self.args)
            mae, mse, r2 = test_metrics_short(true_long, pred_long, self.args.short_pred_length)

            if is_print:
                print('Metrics ood:', self.args.ood_test_traj_num)
                print('----------------------------------------------------------------')
                print(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
                print('----------------------------------------------------------------')
                pred_T = true_long.shape[1]
                print(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.5: {smape_dict[int(pred_T * 0.5)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
                print(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

                test_logger.info('Metrics ood: %d', self.args.ood_test_traj_num)
                test_logger.info('----------------------------------------------------------------')
                test_logger.info(f'VPT: {vpt:.5f}, CD_RMSE: {cd_rmse:.5f}, DH: {dh:.5f}, KLD: {kld:.5f}')
                test_logger.info('----------------------------------------------------------------')
                test_logger.info(f'SMAPE@0.1: {smape_dict[int(pred_T * 0.1)]:.5f}, SMAPE@0.2: {smape_dict[int(pred_T * 0.2)]:.5f}, SMAPE@0.4: {smape_dict[int(pred_T * 0.4)]:.5f}, SMAPE@0.5: {smape_dict[int(pred_T * 0.5)]:.5f}, SMAPE@0.6: {smape_dict[int(pred_T * 0.6)]:.5f}, SMAPE@0.8: {smape_dict[int(pred_T * 0.8)]:.5f}, SMAPE@1.0: {smape_dict[int(pred_T)]:.5f}')
                test_logger.info(f'MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

            filename = f'{self.args.log_dir}/w{self.args.lookback}_t{self.args.data_forecast_length}_ood_{self.args.ood_test_traj_num}_twarm_{self.args.test_warmup_steps}'
            if self.args.test_single_step_only:
                filename += '_stage1'
            with open(filename + '.txt','a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines(f'{random_seed}, {vpt:.5f}, {cd_rmse:.5f}, {dh:.5f}, {kld:.5f}, {mae:.5f}, {mse:.5f}, {r2:.5f}, {smape_dict[int(pred_T * 0.1)]:.5f},{smape_dict[int(pred_T * 0.2)]:.5f}, {smape_dict[int(pred_T * 0.4)]:.5f}, {smape_dict[int(pred_T * 0.5)]:.5f}, {smape_dict[int(pred_T * 0.6)]:.5f}, {smape_dict[int(pred_T * 0.8)]:.5f}, {smape_dict[int(pred_T)]:.5f}\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
