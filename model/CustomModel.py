import math
from collections import defaultdict
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
# Import Huggingface
from transformers import AutoTokenizer, AutoConfig, AutoModel

class CustomModel(nn.Module):
    def __init__(self, encoder_model_type: str = 'bart', decoder_model_type: str = 'bart',
                 src_vocab_num: int = 32000, trg_vocab_num: int = 32000, trg_label_num: int = 2,
                 src_max_len: int = 150, trg_max_len: int = 150,
                 isPreTrain: bool = True, dropout: float = 0.3, pca_comp_n: int = 5, pooling: str = 'first_token'):
        super().__init__()

        """
        Initialize augmenter model

        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device):
        Returns:
            log_prob (torch.Tensor): log probability of each word
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.isPreTrain = isPreTrain
        self.dropout = nn.Dropout(dropout)
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        self.pca_comp_n = pca_comp_n
        self.trg_label_num = trg_label_num
        self.pooling = pooling

        # Encoder model setting
        self.encoder_model_type = encoder_model_type
        encoder_model_name = return_model_name(self.encoder_model_type)
        encoder, encoder_model_config = encoder_model_setting(encoder_model_name, self.isPreTrain)

        self.encoder = encoder

        try:
            self.enc_d_hidden = encoder_model_config.d_model
        except:
            self.enc_d_hidden = encoder_model_config.hidden_size
        self.enc_d_embedding = int(self.enc_d_hidden / 2)

        # Classifier Linear Model Setting
        self.cls_linear = nn.Linear(self.enc_d_hidden, self.enc_d_embedding)
        self.cls_norm = nn.LayerNorm(self.enc_d_embedding, eps=1e-12)
        self.cls_linear2 = nn.Linear(self.enc_d_embedding, self.trg_label_num)

        # Decoder model setting
        self.decoder_model_type = decoder_model_type
        decoder_model_name = return_model_name(self.decoder_model_type)
        decoder, decoder_model_config = decoder_model_setting(decoder_model_name, self.isPreTrain)
        
        try:
            self.dec_d_hidden = decoder_model_config.d_model
        except:
            self.dec_d_hidden = decoder_model_config.hidden_size
        self.dec_d_embedding = int(self.dec_d_hidden / 2)
        self.vocab_num = trg_vocab_num

        self.decoder = decoder

        if self.enc_d_hidden != self.dec_d_hidden:
            self.enc_to_dec_hidden = nn.Linear(self.enc_d_hidden, self.dec_d_hidden)

        # Decoder Linear Model Setting
        self.decoder_linear = nn.Linear(self.dec_d_hidden, self.dec_d_embedding)
        self.decoder_norm = nn.LayerNorm(self.dec_d_embedding, eps=1e-12)
        self.decoder_linear2 = nn.Linear(self.dec_d_embedding, self.vocab_num)

        # Tokenizer Setting
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.pad_idx = self.tokenizer.pad_token_id
        self.decoder_start_token_id = decoder_model_config.decoder_start_token_id
        if self.decoder_model_type == 'bert':
            self.bos_idx = self.tokenizer.cls_token_id
            self.eos_idx = self.tokenizer.sep_token_id
        else:
            self.bos_idx = self.tokenizer.bos_token_id
            self.eos_idx = self.tokenizer.eos_token_id

    def encode(self, src_input_ids, src_attention_mask=None):
        if src_input_ids.dtype == torch.int64:
            encoder_out = self.encoder(input_ids=src_input_ids,
                                       attention_mask=src_attention_mask)
        else:
            encoder_out = self.encoder(inputs_embeds=src_input_ids,
                                       attention_mask=src_attention_mask)
        encoder_out = encoder_out['last_hidden_state'] # (batch_size, seq_len, d_hidden)

        return encoder_out
    
    def pca_reduction(self, encoder_hidden_states):
        U, S, V = torch.pca_lowrank(encoder_hidden_states.transpose(1,2), q=self.pca_comp_n)
        pca_encoder_out = U.transpose(1,2)
        return pca_encoder_out
    
    def decode(self, trg_input_ids, decoder_start_token_id, encoder_hidden_states=None, encoder_attention_mask=None):
        decoder_input_ids = shift_tokens_right(
            trg_input_ids, self.pad_idx, decoder_start_token_id
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        return decoder_outputs['last_hidden_state']
    
    def classify(self, hidden_states, eos_mask=None):

        if self.pooling == 'first_token':
            encoder_outputs = hidden_states[:,0,:] # (batch_size, d_hidden)
        if self.pooling == 'last_token':
            hidden_states = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))
            encoder_outputs = hidden_states[:,-1,:] # (batch_size, d_hidden)
        if self.pooling == 'max':
            encoder_outputs = hidden_states.max(dim=1)[0] # (batch_size, d_hidden)
        if self.pooling == 'mean':
            encoder_outputs = hidden_states.mean(dim=1) # (batch_size, d_hidden)

        encoder_outputs = self.dropout(F.gelu(self.cls_linear(encoder_outputs)))
        logits = self.cls_linear2(self.cls_norm(encoder_outputs))

        return logits

    def generate(self, hidden_states):

        hidden_states = self.dropout(F.gelu(self.decoder_linear(hidden_states)))
        hidden_states = self.decoder_linear2(self.decoder_norm(hidden_states))

        return hidden_states
    
    def sampling(self, decoding_dict:dict = dict(), decoder_start_token_id: int = 0, encoder_hidden_states=None, encoder_attention_mask=None):
        
        if decoding_dict['decoding_strategy'] == 'beam':
            # Input, output setting
            device = encoder_hidden_states.device
            batch_size = encoder_hidden_states.size(0)
            src_seq_size = encoder_hidden_states.size(1)
        
            # Decoding dictionary
            beam_size = decoding_dict['beam_size']
            beam_alpha = decoding_dict['beam_alpha']
            repetition_penalty = decoding_dict['repetition_penalty']

            # Encoder hidden state expanding
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1) # (batch_size, 1, seq_len, d_hidden)
            encoder_hidden_states = encoder_hidden_states.repeat(1, beam_size, 1, 1) # (batch_size, beam_size, seq_len, d_hidden)
            encoder_hidden_states = encoder_hidden_states.view(-1, src_seq_size, self.dec_d_hidden) # (batch_size * beam_size, seq_len, d_hidden)

            if not encoder_attention_mask == None:
                encoder_attention_mask = encoder_attention_mask.view(batch_size, 1, -1)
                encoder_attention_mask = encoder_attention_mask.repeat(1, beam_size, 1)
                encoder_attention_mask = encoder_attention_mask.view(-1, src_seq_size)

            # Scores save vector & decoding list setting
            scores_save = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * beam_size, 1)
            top_k_scores = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * beam_size, 1)
            every_batch = torch.arange(0, beam_size * batch_size, beam_size, device=device)
            complete_seqs = defaultdict(list)
            complete_ind = set()

            # Decoding start token setting
            seqs = torch.tensor([[decoder_start_token_id]], dtype=torch.long, device=device)
            seqs = seqs.repeat(beam_size * batch_size, 1).contiguous() # (batch_size * beam_size, 1)

            for step in range(self.trg_max_len):
                # Decoding sentence
                decoder_outputs = self.decoder(
                    input_ids=seqs,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
                decoder_outputs = decoder_outputs['last_hidden_state']

                # Score calculate
                scores = F.gelu(self.decoder_linear(decoder_outputs[:,-1])) # (batch_size * k, d_embedding)
                scores = self.decoder_linear2(self.decoder_norm(scores)) # (batch_size * k, vocab_num)
                scores = scores / decoding_dict['softmax_temp']
                scores = F.softmax(scores, dim=1) # (batch_size * k, vocab_num)

                # Add score
                scores = top_k_scores.expand_as(scores) + scores  # (batch_size * k, vocab_num)
                if step == 0:
                    scores = scores[::beam_size] # (batch_size, vocab_num)
                    scores[:, self.eos_idx] = float('-inf') # set eos token probability zero in first step
                    top_k_scores, top_k_words = scores.topk(beam_size, 1, True, True)  # (batch_size, k) , (batch_size, k)
                else:
                    top_k_scores, top_k_words = scores.view(batch_size, -1).topk(beam_size, 1, True, True)

                # Previous and Next word extract
                prev_word_inds = top_k_words // self.vocab_num # (batch_size * k, out_seq)
                next_word_inds = top_k_words % self.vocab_num # (batch_size * k, out_seq)
                top_k_scores = top_k_scores.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
                top_k_words = top_k_words.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
                seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, beam_size).view(-1)] # (batch_size * k, out_seq)
                seqs = torch.cat([seqs, next_word_inds.view(beam_size * batch_size, -1)], dim=1) # (batch_size * k, out_seq + 1)

                # Find and Save Complete Sequences Score
                if self.eos_idx in next_word_inds:
                    eos_ind = torch.where(next_word_inds.view(-1) == self.eos_idx)
                    eos_ind = eos_ind[0].tolist()
                    complete_ind_add = set(eos_ind) - complete_ind
                    complete_ind_add = list(complete_ind_add)
                    complete_ind.update(eos_ind)
                    if len(complete_ind_add) > 0:
                        scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                        for ix in complete_ind_add:
                            complete_seqs[ix] = seqs[ix].tolist()

            # If eos token doesn't exist in sequence
            if 0 in scores_save:
                score_save_pos = torch.where(scores_save == 0)
                for ix in score_save_pos[0].tolist():
                    complete_seqs[ix] = seqs[ix].tolist()
                scores_save[score_save_pos] = top_k_scores[score_save_pos]

            # Beam Length Normalization
            lp = torch.tensor([len(complete_seqs[i]) for i in range(batch_size * beam_size)], device=device)
            lp = (((lp + beam_size) ** beam_alpha) / ((beam_size + 1) ** beam_alpha)).unsqueeze(1)
            scores_save = scores_save / lp

            # Predicted and Label processing
            _, ind = scores_save.view(batch_size, beam_size, -1).max(1)
            ind_expand = ind.view(-1) + every_batch
            predicted = [complete_seqs[i] for i in ind_expand.tolist()]

            return predicted

        else:
            raise Exception('Comming Soon...')

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def encoder_model_setting(model_name, isPreTrain):
    model_config = AutoConfig.from_pretrained(model_name)

    if isPreTrain:
        basemodel = AutoModel.from_pretrained(model_name)
    else:
        basemodel = AutoModel.from_config(model_config)

    if model_name == 'bert-large-cased':
        encoder = basemodel
    else:
        encoder = basemodel.encoder

    return encoder, model_config

def decoder_model_setting(model_name, isPreTrain):
    model_config = AutoConfig.from_pretrained(model_name)

    if isPreTrain:
        basemodel = AutoModel.from_pretrained(model_name)
    else:
        basemodel = AutoModel.from_config(model_config)

    decoder = basemodel.decoder

    return decoder, model_config

def return_model_name(model_type):
    if model_type == 'bert':
        out = 'bert-large-cased'
    if model_type == 'albert':
        out = 'albert-base-v2'
    if model_type == 'deberta':
        out = 'microsoft/deberta-v3-base'
    if model_type == 'bart':
        out = 'facebook/bart-base'
    if model_type == 'kr_bart':
        out = 'cosmoquester/bart-ko-mini'
    if model_type == 'T5':
        out = 't5-large'
    return out