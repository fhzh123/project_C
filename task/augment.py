import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# Custom Modules
from model.CustomModel import CustomModel, return_model_name
from model.dataset import ReconDataset
from model.optimizer.utils import optimizer_select, scheduler_select
from task.utils import TqdmLoggingHandler, write_log
from data.data_load import data_load

def augment(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, "Load data...")

    total_src_list, total_trg_list = data_load(data_path=args.data_path, data_name=args.data_name)

    # tokenizer load
    src_tokenizer_name = return_model_name(args.src_tokenizer_type)
    src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
    src_vocab_num = src_tokenizer.vocab_size

    trg_tokenizer_name = return_model_name(args.trg_tokenizer_type)
    trg_tokenizer = AutoTokenizer.from_pretrained(trg_tokenizer_name)
    trg_vocab_num = trg_tokenizer.vocab_size

    dataset_dict = {
        'train': ReconDataset(src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer, src_list=total_src_list['train'], trg_list=total_trg_list['train'], 
                              src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True, batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    model = CustomModel(encoder_model_type=args.encoder_model_type, decoder_model_type=args.decoder_model_type,
                        src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                        trg_label_num=len(set(total_trg_list['train'])),
                        src_max_len=args.src_max_len, isPreTrain=args.isPreTrain, dropout=args.dropout, 
                        pooling=args.pooling, pca_comp_n=args.pca_comp_n)
    save_file_name = f'checkpoint_enc_{args.encoder_model_type}_dec_{args.decoder_model_type}_pca_{args.pca_reduction}_seed_{args.random_seed}.pth.tar'
    save_file_path = os.path.join(args.model_save_path, args.data_name, save_file_name)
    checkpoint = torch.load(save_file_path)
    model.load_state_dict(checkpoint['model'])
    write_log(logger, f'Loaded augmenter model from {save_file_path}')
    model.to(device)

    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(optimizer_model=args.optimizer, model=model, lr=args.lr, w_decay=args.w_decay)
    cudnn.benchmark = True
    cls_criterion = nn.CrossEntropyLoss().to(device)
    recon_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps, ignore_index=model.pad_idx).to(device)

    #===================================#
    #===========Augment Start===========#
    #===================================#

    write_log(logger, 'Augment start!')

    start_time_e = time()

    origin_list = list()
    origin_label_list = list()
    recon_list = list()
    new_recon_list = list()
    new_beam_recon_list = list()
    moved_label_list = list()

    for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

        model.zero_grad(set_to_none=True)

        # Input setting
        src_sequence = batch_iter[0][0].to(device, non_blocking=True)
        src_att = batch_iter[0][1].to(device, non_blocking=True)

        trg_sequence = batch_iter[1][0].to(device, non_blocking=True) # Same as src_sequence
        trg_sequence = trg_sequence[:,1:,]
        _ = batch_iter[1][1].to(device, non_blocking=True)
        trg_label = batch_iter[1][2].to(device, non_blocking=True)
        fliped_trg_label = torch.flip(F.one_hot(torch.tensor(trg_label, dtype=torch.long)), dims=[1]).to(device)

        # Encoding
        encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_att)

        # PCA
        if args.pca_reduction:
            encoder_out = model.pca_reduction(hidden_states=encoder_out, encoder_attention_mask=src_att)
            src_att = None

        # Decoding
        hidden_states = model.decode(trg_input_ids=trg_sequence, decoder_start_token_id=model.decoder_start_token_id, 
                                    encoder_hidden_states=encoder_out, encoder_attention_mask=src_att)
        new_recon = model.generate(hidden_states=hidden_states)
        recon_text_token = new_recon.argmax(dim=2)
        recon_text = trg_tokenizer.batch_decode(recon_text_token)

        # Label Flipping
        encoder_out_grad_true = encoder_out.clone().detach().requires_grad_(True)

        # Classify
        eos_mask = src_sequence.eq(model.eos_idx).to(encoder_out.device)
        logit = model.classify(hidden_states=encoder_out_grad_true, eos_mask=eos_mask)

        cls_loss = cls_criterion(logit, fliped_trg_label.float())
        cls_loss.backward()

        hidden_states_grad = encoder_out_grad_true.grad.data.sign()
        encoder_out_copy = encoder_out.clone().detach()
        latent_out = encoder_out_copy - (args.fgsm_epsilon * hidden_states_grad)

        new_logit = model.classify(hidden_states=latent_out, eos_mask=eos_mask)
        new_trg_label = new_logit.argmax(dim=1).tolist()

        with torch.no_grad():
            hidden_states = model.decode(trg_input_ids=trg_sequence, decoder_start_token_id=model.decoder_start_token_id, 
                                        encoder_hidden_states=latent_out, encoder_attention_mask=src_att)
            recon = model.generate(hidden_states=hidden_states)

            flip_recon_text_token = recon.argmax(dim=2)
            flip_recon_text = trg_tokenizer.batch_decode(flip_recon_text_token)

            decoding_dict = {
                'decoding_strategy': 'beam',
                'beam_size': 5,
                'beam_alpha': 0.7,
                'repetition_penalty': 0.7,
                'softmax_temp': 0.9
            }
            beam_results_ = model.sampling(decoding_dict=decoding_dict, decoder_start_token_id=model.decoder_start_token_id,
                                           encoder_hidden_states=latent_out, encoder_attention_mask=src_att)
            beam_results = trg_tokenizer.batch_decode(beam_results_)
        

        origin_list.extend(trg_tokenizer.batch_decode(trg_sequence))
        origin_label_list.extend(trg_label.tolist())
        recon_list.extend(recon_text)
        new_recon_list.extend(flip_recon_text)
        new_beam_recon_list.extend(beam_results)
        moved_label_list.extend(new_trg_label)

        if i == 10:
            break

    augmented_dat = pd.DataFrame({
        'origin_text': origin_list,
        'origin_label': origin_label_list,
        'recon_text': recon_list,
        'recon_label' : origin_label_list,
        'new_recon_text': new_recon_list,
        'new_beam_recon_list': new_beam_recon_list,
        'new_recon_label': moved_label_list,
    })
    augmented_dat.to_csv(f'test_{args.random_seed}.csv', index=False)