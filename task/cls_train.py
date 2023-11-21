import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import numpy as np
from tqdm import tqdm
from time import time
# Import PyTorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# Custom Modules
from model.CustomModel import CustomModel, return_model_name
from model.dataset import Seq2LabelDataset
from model.optimizer.utils import optimizer_select, scheduler_select
from task.utils import TqdmLoggingHandler, write_log
from data.data_load import data_load

def cls_training(args):

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
        'train': Seq2LabelDataset(src_tokenizer=src_tokenizer, src_list=total_src_list['train'], trg_list=total_trg_list['train'], 
                                  src_max_len=args.src_max_len),
        'valid': Seq2LabelDataset(src_tokenizer=src_tokenizer, src_list=total_src_list['valid'], trg_list=total_trg_list['valid'], 
                                  src_max_len=args.src_max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True, batch_size=args.batch_size, shuffle=True,
                            pin_memory=True, num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], drop_last=True, batch_size=args.batch_size, shuffle=True, 
                            pin_memory=True, num_workers=args.num_workers)
    }
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    model = CustomModel(encoder_model_type=args.encoder_model_type, decoder_model_type=args.decoder_model_type,
                        src_vocab_num=src_vocab_num, trg_vocab_num=trg_vocab_num,
                        trg_label_num=len(set(total_trg_list['train'])),
                        src_max_len=args.src_max_len, isPreTrain=args.isPreTrain, dropout=args.dropout, 
                        pooling=args.pooling, pca_comp_n=args.pca_comp_n)
    model.to(device)

    # 3) Optimizer & Learning rate scheduler setting
    optimizer = optimizer_select(optimizer_model=args.optimizer, model=model, lr=args.lr, w_decay=args.w_decay)
    scheduler = scheduler_select(scheduler_model=args.scheduler, optimizer=optimizer, dataloader_len=len(dataloader_dict['train']), task='cls', args=args)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing_eps).to(device)

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        write_log(logger, 'Resume model...')
        save_file_name = f'checkpoint_enc_{args.encoder_model_type}_dec_{args.decoder_model_type}_pca_{args.pca_reduction}_seed_{args.random_seed}.pth.tar'
        save_file_path = os.path.join(args.model_save_path, args.data_name, save_file_name)
        checkpoint = torch.load(save_file_path)
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Traing start!')
    best_val_loss = 1e+7

    for epoch in range(start_epoch + 1, args.cls_num_epochs + 1):
        start_time_e = time()

        write_log(logger, 'Training start...')
        model.train()

        for i, batch_iter in enumerate(tqdm(dataloader_dict['train'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}')):

            optimizer.zero_grad(set_to_none=True)

            # Input setting
            src_sequence = batch_iter[0][0].to(device, non_blocking=True)
            src_att = batch_iter[0][1].to(device, non_blocking=True)
            trg_label = batch_iter[1].to(device, non_blocking=True)

            # Encoding
            encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_att)

            # PCA
            if args.pca_reduction:
                encoder_out = model.pca_reduction(encoder_hidden_states=encoder_out)
                src_att = None

            # Classify
            eos_mask = src_sequence.eq(model.eos_idx).to(encoder_out.device)
            logit = model.classify(hidden_states=encoder_out, eos_mask=eos_mask)
            
            # Loss Backward
            train_loss = criterion(logit, trg_label)
            train_loss.backward()
            if args.clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            # Print loss value only training
            if i == 0 or i % args.print_freq == 0 or i == len(dataloader_dict['train'])-1:
                train_acc = (logit.argmax(dim=1) == trg_label).sum() / len(trg_label)
                iter_log = "[Epoch:%03d][%03d/%03d] train_loss:%03.2f | train_accuracy:%03.2f | learning_rate:%1.6f |spend_time:%02.2fmin" % \
                    (epoch, i, len(dataloader_dict['train'])-1, train_loss.item(), train_acc.item() * 100, optimizer.param_groups[0]['lr'], (time() - start_time_e) / 60)
                write_log(logger, iter_log)

            if args.debugging_mode:
                break

        write_log(logger, 'Validation start...')
        model.eval()
        val_loss = 0
        val_acc = 0

        for batch_iter in tqdm(dataloader_dict['valid'], bar_format='{l_bar}{bar:30}{r_bar}{bar:-2b}'):

            # Input setting
            src_sequence = batch_iter[0][0].to(device, non_blocking=True)
            src_att = batch_iter[0][1].to(device, non_blocking=True)
            trg_label = batch_iter[1].to(device, non_blocking=True)

            with torch.no_grad():

                # Encoding
                encoder_out = model.encode(src_input_ids=src_sequence, src_attention_mask=src_att)

                # PCA
                if args.pca_reduction:
                    encoder_out = model.pca_reduction(encoder_hidden_states=encoder_out)
                    src_att = None

                # Classify
                eos_mask = src_sequence.eq(model.eos_idx).to(encoder_out.device)
                logit = model.classify(hidden_states=encoder_out, eos_mask=eos_mask)

            val_acc += (logit.argmax(dim=1) == trg_label).sum() / len(trg_label)
            val_loss += criterion(logit, trg_label)

            if args.debugging_mode:
                break

        # val_mmd_loss /= len(dataloader_dict['valid'])
        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation CrossEntropy Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.2f%%' % (val_acc * 100))

        save_file_name = f'checkpoint_enc_{args.encoder_model_type}_dec_{args.decoder_model_type}_pca_{args.pca_reduction}_seed_{args.random_seed}.pth.tar'
        save_file_path = os.path.join(args.model_save_path, args.data_name, save_file_name)
        if val_loss < best_val_loss:
            write_log(logger, 'Model checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_file_path)
            best_val_loss = val_loss
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch Loss({round(best_val_loss.item(), 5)}) is better...'
            write_log(logger, else_log)