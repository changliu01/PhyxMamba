import argparse

def get_args():

    parser = argparse.ArgumentParser()
    ### exp_settings
    parser.add_argument('--exp_name', type=str, default='standard')
    parser.add_argument('--multi_num', type=int, default=20)
    parser.add_argument('--relative_times_standard', type=int, default=1)
    
    parser.add_argument('--ood_test_traj_num', type=int, default=20)
    parser.add_argument('--test_ood', action='store_true')
    parser.add_argument('--random_seed', type=int, default=729)
    parser.add_argument('--seed_num', type=int, default=1)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--cpu_num', type=int, default=1)
    parser.add_argument('--memory_size', type=int, default=6500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir_base', type=str, default='./data')
    parser.add_argument('--system', type=str, default='Lorenz')
    parser.add_argument('--atol', type=float, default=1e-12)
    parser.add_argument('--rtol', type=float, default=1e-12)
    parser.add_argument('--kld_upper_bound', type=float, default=40.0)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--final_data_ratio', type=float, default=1.0)

    parser.add_argument('--print_metrics_dir', type=str, default='./results')
    parser.add_argument('--exp_type', type=str, default='others')

    parser.add_argument('--partial_observe', type=int, default=0)
    parser.add_argument('--partial_observe_dim', type=int, default=2)


    parser.add_argument('--data_file_path', type=str, default='./data')
    parser.add_argument('--skip_stage1', action='store_true')
    parser.add_argument('--skip_stage2', action='store_true')

    parser.add_argument('--augmentation_ratio', type=float, default=0.0)
    
    parser.add_argument('--no_check_data', type=int, default=0)
    parser.add_argument('--shift_dataset', type=int, default=0)
    parser.add_argument('--num_trajectory', type=int, default=20)

    parser.add_argument('--large_train', type=int, default=0)
    parser.add_argument('--large_train_ratio', type=float, default=1)
    parser.add_argument('--validation', type=int, default=0)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--val_long', type=int, default=0)
    parser.add_argument('--multi_init_test', action='store_true')
    parser.add_argument('--test_single_step_only', action='store_true')

    parser.add_argument('--data_training_length', type=int, default=-1)
    parser.add_argument('--training_ratio', type=float, default=5/6)
    parser.add_argument('--split_point', type=int, default=-1)
    parser.add_argument('--data_valid_length', type=int, default=-1)
    parser.add_argument('--data_forecast_length', type=int, default=-1)
    parser.add_argument('--short_pred_length', type=int, default=1)
    parser.add_argument('--parameter_tuning_data', type=int, default=0)
    parser.add_argument('--num_trajectory_tune', type=int, default=1)
    parser.add_argument('--search_hyper_parameter', type=int, default=1)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    
    parser.add_argument('--use_revin', type=int, default=0)

    # parser.add_argument('--data_forecast_context_length', type=int, default=512)
    parser.add_argument('--steps_per_traj', type=int, default=1000)
    parser.add_argument('--pts_per_period', type=int, default=30)
    parser.add_argument('--traj_length', type=int, default=812)
    parser.add_argument('--horizon_tr', type=int, default=1)
    parser.add_argument('--horizon_te', type=int, default=300)
    parser.add_argument('--horizon_multistep', type=int, default=30)
    parser.add_argument('--data_ratio', type=float, default=1.0)
    parser.add_argument('--multi_step_data_ratio', type=float, default=1.0)
    parser.add_argument('--lookback', type=int, default=512)

    parser.add_argument('--nhits_num_stacks', type=int, default=5)
    parser.add_argument('--nhits_num_blocks', type=int, default=5)
    
    parser.add_argument('--generate_traj_num', type=int, default=20)
    parser.add_argument('--allow_gpu_list', type=str, default='[1]')
    parser.add_argument('--model', type=str, default='chronos_zeroshot')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=1600)
    parser.add_argument('--stage1_max_epoch',type=int, default=800)
    parser.add_argument('--stage2_linear_epoch', type=int, default=200)
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--get_test_time_only', type=int, default=0)


    
    parser.add_argument('--test_debug', action='store_true')
    parser.add_argument('--data_generation_only', action='store_true')
    parser.add_argument('--importance_sampling_epoch', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--meta_epochs', type=int, default=100)
    parser.add_argument('--importance_sampling', type=int, default=0)
    parser.add_argument('--importance_sampling_rate', type=float, default=0.1)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--invariant_weight_decay', type=float, default=1e-5)
    parser.add_argument('--contrastive_val_interval', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--long_test_length', type=int, default=300)
    
    # data L96
    parser.add_argument('--generalization', type=int, default=0)
    parser.add_argument('--F_min', type=float, default=12.0)
    parser.add_argument('--F_max', type=float, default=18.0)
    parser.add_argument('--system_dim', type=int, default=20)
    # parser.add_argument('--num_of_samples', type=int, default=2000)
    parser.add_argument('--total_env_num', type=int, default=2000)
    # parser.add_argument('--use_envs', type=int, default=2000)
    parser.add_argument('--use_env_ratio', type=float, default=0.1)
    parser.add_argument('--rho_min', type=float, default=25)
    parser.add_argument('--rho_max', type=float, default=40)
    parser.add_argument('--rho_min_train', type=float, default=25)
    parser.add_argument('--rho_max_train', type=float, default=40)
    parser.add_argument('--rho_min_test', type=float, default=25)
    parser.add_argument('--rho_max_test', type=float, default=40)
    
    # Chronos
    parser.add_argument('--model_type', type=str, default='tiny')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--chronos_hidden_size', type=int, default=64)
    
    parser.add_argument('--lambda_dis', type=float, default=1e-3)
    parser.add_argument('--lambda_fnn', type=float, default=1.0)
    parser.add_argument('--n_layers', type=int, default=2)

    # iTransformer
    parser.add_argument('--use_norm', type=int, default=0)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h', help='frequency of data')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--e_layers', type=int, default=2)
    
    # Autoformer
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

    # RC
    parser.add_argument('--warm_up', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--n_internal_units', type=int, default=1000)
    parser.add_argument('--spectral_radius', type=float, default=0.85)
    parser.add_argument('--leak', type=float, default=0.05)
    parser.add_argument('--connectivity', type=float, default=0.2)
    parser.add_argument('--input_scaling', type=float, default=0.1)
    parser.add_argument('--input_connectivity', type=float, default=0.2)
    parser.add_argument('--noise_level', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--direc', type=int, default=1)
    parser.add_argument('--node_num', type=int, default=1)
    parser.add_argument('--net_nam', type=str, default='er')

    #NVAR
    parser.add_argument('--nvar_delay', type=int, default=100)
    parser.add_argument('--nvar_order', type=int, default=1)
    parser.add_argument('--nvar_strides', type=int, default=1)

    # NBEATS
    parser.add_argument('--interpret', type=int, default=0)
    parser.add_argument('--degree_of_polynomial', type=int, default=3)
    parser.add_argument('--trend_layers', type=int, default=4)
    parser.add_argument('--trend_layer_size', type=int, default=256)
    parser.add_argument('--num_of_harmonics', type=int, default=1)
    parser.add_argument('--seasonality_layers', type=int, default=4)
    parser.add_argument('--seasonality_layer_size', type=int, default=256)
    parser.add_argument('--trend_blocks', type=int, default=3)
    parser.add_argument('--seasonality_blocks', type=int, default=3)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--layer_size', type=int, default=256)
    parser.add_argument('--stacks', type=int, default=30)

    # Mamba
    parser.add_argument('--enc_layers', type=int, default=2)
    parser.add_argument('--mamba_d_state', type=int, default=128)
    parser.add_argument('--mamba_d_conv', type=int, default=4)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_headdim', type=int, default=64)
    parser.add_argument('--data_emb_conv_kernel', type=int, default=3)
    parser.add_argument('--use_time_delay', type=int, default=0)
    parser.add_argument('--use_backward_model', type=int, default=0)
    parser.add_argument('--delay', type=int, default=4)
    parser.add_argument('--delay_embedding_size', type=int, default=384)
    parser.add_argument('--flatten_output', type=int, default=1)
    parser.add_argument('--small_projection_size', type=int, default=4)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--time_conv_prediction', type=int, default=0)
    parser.add_argument('--pad_prediction', type=int, default=0)
    parser.add_argument('--pad_before', type=int, default=0)
    # parser.add_argument('--contrastive_batch_size', type=int, default=32)
    parser.add_argument('--contrastive_max_epoch', type=int, default=100)
    parser.add_argument('--invariant_emb_size', type=int, default=32)
    parser.add_argument('--invariant_batch_size', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--negative_weight', type=float, default=1)
    parser.add_argument('--invariant_lr', type=float, default=1e-3)
    parser.add_argument('--invariant_val_interval', type=int, default=10)
    parser.add_argument('--n_envs', type=int, default=10)
    parser.add_argument('--commit_loss_alpha', type=float, default=0.5)
    parser.add_argument('--vq_loss_alpha', type=float, default=1)
    parser.add_argument('--invariant_loss_alpha', type=float, default=1)
    parser.add_argument('--n_decoder_layers', type=int, default=1)
    parser.add_argument('--env_in_final', type=int, default=0)

    parser.add_argument('--patch_model_method', type=str, default='')
    parser.add_argument('--embedding_method', type=str, default='default')
    parser.add_argument('--delay_emb_dim', type=int, default=3)
    parser.add_argument('--delay_tau', type=int, default=5)
    parser.add_argument('--decoder_dropout', type=float, default=0)

    parser.add_argument('--hier_layers', type=int, default=0)
    parser.add_argument('--mtp_steps', type=int, default=3)
    parser.add_argument('--lambda_mtp_loss', type=float, default=1e-1)

    parser.add_argument('--lambda_uncond_mmd', type=float, default=0)
    parser.add_argument('--lambda_cond_mmd', type=float, default=0)

    parser.add_argument('--mamba_layers', type=int, default=4)



    parser.add_argument('--hopfield_hidden_size', type=int, default=512)
    parser.add_argument('--hopfield_num_heads', type=int, default=8)
    parser.add_argument('--hopfield_num_patterns', type=int, default=256)
    parser.add_argument('--lambda_hopfield_loss', type=float, default=1e-1)

    parser.add_argument('--context_dim', type=int, default=3)
    parser.add_argument('--contrastive_batch_per_epoch', type=int, default=100)
    parser.add_argument('--contrastive_batch_size', type=int, default=32)
    parser.add_argument('--contrastive_noise_level', type=float, default=0.0)

    parser.add_argument('--save_intermediate', action='store_true')

    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=5)

    parser.add_argument('--use_moco', type=int, default=0)
    parser.add_argument('--contrastive_weight', type=float, default=10.)
    parser.add_argument('--momentum_encoder', type=int, default=1)
    
    parser.add_argument('--use_codebook', type=int, default=1)
    parser.add_argument('--use_in_encoding', type=int, default=1)
    parser.add_argument('--use_in_decoding', type=int, default=1)
    
    parser.add_argument('--env_emb_v2', type=int, default=1)

    parser.add_argument('--attention_decoding', type=int, default=1)

    parser.add_argument('--time_aware_pooling', type=int, default=0)
    parser.add_argument('--var_aware_pooling', type=int, default=0)

    parser.add_argument('--mix_dataset', type=int, default=0)

    parser.add_argument('--hard_code_env', type=int, default=0)

    parser.add_argument('--specify_ckpt', type=int, default=0)

    parser.add_argument('--finetune_ckpt_path', type=str, default='')
    parser.add_argument('--finetune_val_interval', type=int, default=10)
    parser.add_argument('--finetune_save_interval', type=int, default=10)
    # parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--T', type=float, default=1000)
    parser.add_argument('--d_intermediate', type=int, default=512)
    parser.add_argument('--fused_add_norm', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=1024) # approximate 0.01 * 1024 = 10 in absolute timescale
    parser.add_argument('--token_size', type=int, default=1)
    parser.add_argument('--test_warmup_steps', type=int, default=10)
    parser.add_argument('--start_idx', type=int, default=15000)
    parser.add_argument('--long_term_pred_steps', type=int, default=15000)
    parser.add_argument('--short_term_pred_steps', type=int, default=20)
    # parser.add_argument('--ood', type=int, default=1)
    parser.add_argument('--indis_multi_env', type=int, default=1)
    parser.add_argument('--test_context_size', type=int, default=100)
    parser.add_argument('--adjust_resolution', type=int, default=0)
    parser.add_argument('--sample_data', type=int, default=1)
    
    parser.add_argument('--delay_cut', type=int, default=0)
    parser.add_argument('--temporal_memory', type=int, default=1)
    parser.add_argument('--num_memory', type=int, default=512)
    
    parser.add_argument('--fft_mode', type=int, default=1)
    parser.add_argument('--fft_thred', type=int, default=0)
    parser.add_argument('--freq_memory', type=int, default=1)
    parser.add_argument('--lambda_time_rec', type=float, default=1e-2)
    parser.add_argument('--lambda_freq_rec', type=float, default=1e-2)

    parser.add_argument('--contrastive_pretrain', type=int, default=0)
    parser.add_argument('--pretrain_batch_size', type=int, default=128)
    parser.add_argument('--pretrain_batches_per_epoch', type=int, default=1)
    parser.add_argument('--pretrain_max_epochs', type=int, default=100)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--temporal_prompt', type=int, default=0)
    parser.add_argument('--flip_conv', type=int, default=1)

    parser.add_argument('--invariant_supervise_type', type=str, default='time-wise')
    parser.add_argument('--finetune_epochs', type=int, default=0)
    parser.add_argument('--finetune_fracs', type=float, default=0.0)
    parser.add_argument('--from_scratch', type=int, default=0)
    
    # parser.add_argument('--linear_probing', type=int, default=1)
    # parser.add_argument('--finetune_memory', type=int, defa/ult=1)
    # parser.add_argument('--linear_probing_epochs', type=int, default=100)
    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--finetune_lr', type=float, default=1e-3)
    # parser.add_argument('--finetune_lr_phase2', type=float, default=1e-5)

    parser.add_argument('--external_parallel_gpu', type=int, default=0)
    parser.add_argument('--model_size', type=str, default='use_parameter')

    ## for ablation studies
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--CD', type=int, default=1)
    parser.add_argument('--use_inv_loss', type=int, default=1)
    parser.add_argument('--exp_mark', type=str, default='')

    ## training setting
    parser.add_argument('--train_from_ckpts', type=int, default=0)
    parser.add_argument('--ckpts_epoch', type=int, default=0)
    parser.add_argument('--retrain_lr', type=float, default=1e-5)

    # test setting
    parser.add_argument('--load_best', type=int, default=0)

    args = parser.parse_args()

    return args