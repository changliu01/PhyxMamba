
def get_suffix(args):

    suffix = f'lookback{args.lookback}_plen{args.horizon_tr}'

    if args.model == 'ours':
                
        if args.use_time_delay:
            suffix = f'delayd{args.delay_emb_dim}_tau{args.delay_tau}_' + suffix

            if args.delay_cut:
                suffix = 'cut_' + suffix

        if args.temporal_memory:
            
                suffix = f'enc_{args.use_in_encoding}_dec_{args.use_in_decoding}_tmem_{args.num_memory}_' + suffix

                if args.temporal_prompt:
                    suffix = 'temprompt_' + suffix
                
                if args.flip_conv:
                    suffix = 'flipconv_' + suffix
        
        if args.freq_memory:
            suffix = f'enc_{args.use_in_encoding}_dec_{args.use_in_decoding}_fmem_{args.num_memory}_fmode_{args.fft_mode}_fthred_{args.fft_thred}_' + suffix

        if args.temporal_memory or args.freq_memory:

            suffix = f'{args.invariant_supervise_type}_' + suffix

        if args.CD:
            suffix = 'CD_' + suffix

        if args.ablation:
            suffix = 'ablation_' + suffix

        suffix = f'{args.model_size}_d_{args.d_model}_n_{args.n_layers}_' + suffix

        if args.exp_mark != '':
            suffix = f'{args.exp_mark}_' + suffix
        
        if args.shift_dataset:
            suffix = f'shift_{args.horizon_tr}_' + suffix

        if args.large_train:
            suffix = f'large_ratio{args.large_train_ratio:.3f}_' + suffix
        if args.split_point != -1:
            suffix = f'split_point{args.split_point}_' + suffix
            
    else:
        if args.large_train:
            suffix = f'large_ratio{args.large_train_ratio:.3f}_' + suffix
        if args.split_point != -1:
            suffix = f'split_point{args.split_point}_' + suffix

    suffix = f'CD_{args.CD}_' + suffix
    
    if args.model == 'ours' or args.model == 'Mamba' or args.model == 'Mamba_Hopfield' or args.model == 'Mamba_MTP':
        suffix = f'data_emb_conv_kernel_{args.data_emb_conv_kernel}_' + suffix
        suffix = f'd_model{args.d_model}_d_state{args.mamba_d_state}_expand{args.mamba_expand}_headdim{args.mamba_headdim}_' + suffix
    suffix = f'data_ratio{args.data_ratio}_' + suffix
    
    if args.model == 'ours' or args.model == 'Mamba' or args.model == 'Mamba_Hopfield' or args.model == 'Mamba_MTP':
        suffix = f'mt{args.horizon_multistep}_mep{args.max_epoch}_s1ep{args.stage1_max_epoch}_s2lep{args.stage2_linear_epoch}_' + suffix
        if args.model == 'Mamba_Hopfield':
            suffix = f'lambda_hopf{args.lambda_hopfield_loss}_hophid{args.hopfield_hidden_size}_hopnhead{args.hopfield_num_heads}_' + suffix
        
        if args.use_revin:
            suffix = f'revin_' + suffix
        if args.use_norm:
            suffix = f'norm_' + suffix

        if args.token_size != 1:
            suffix = f'token{args.token_size}_' + suffix
            suffix = f'patch_model_{args.patch_model_method}_' + suffix

        if args.embedding_method != 'default':
            if args.embedding_method == 'psr':
                suffix = f'psr_m_{args.delay_emb_dim}_tau_{args.delay_tau}' + suffix
                if args.decoder_dropout > 0:
                    suffix = f'dropout_{args.decoder_dropout}_' + suffix
            else:
                suffix = f'embedding_{args.embedding_method}_' + suffix

        if args.hier_layers > 0:
            suffix = f'hi{args.hier_layers}_' + suffix
        else:
            suffix = f'nohil{args.mamba_layers}_' + suffix

        if args.mtp_steps > 0:
            suffix = f'mtp{args.mtp_steps}_lb_{args.lambda_mtp_loss}_' + suffix
        else:
            if args.model == 'Mamba_MTP':
                suffix = f'nomtp_' + suffix

        if args.lambda_uncond_mmd > 0 or args.lambda_cond_mmd > 0:
            suffix = f'{args.lambda_uncond_mmd}_{args.lambda_cond_mmd}_' + suffix


        suffix = f'{args.exp_name}_' + suffix

        if args.noise > 0.0:
            suffix = f'noise_{args.noise}_' + suffix

        if args.final_data_ratio < 1.0:
            suffix = f'fdr{args.final_data_ratio}_' + suffix

    if args.model == 'Context':
        suffix = f'd_context{args.context_dim}_d_model{args.d_model}_hop_hidden{args.hopfield_hidden_size}_hophead{args.hopfield_num_heads}_hoppat{args.hopfield_num_patterns}_epoch{args.max_epoch}' + suffix
        suffix = f'{args.exp_name}_' + suffix

    if args.model == 'NBEATS' or args.model == 'NHiTS':

        suffix = f'test_warmup_steps{args.test_warmup_steps}_' + suffix
    
    if args.model == 'NHiTS':
        suffix = f'n_stacks{args.nhits_num_stacks}_n_blocks{args.nhits_num_blocks}_' + suffix

    if args.model == 'ESN':

        suffix = f'leak{args.leak}_' + suffix

    if args.model == 'NVAR':

        suffix = f'delay{args.nvar_delay}_order{args.nvar_order}_strides{args.nvar_strides}_' + suffix
    
    
    return suffix