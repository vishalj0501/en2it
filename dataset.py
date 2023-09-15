import torch
import torch.nn as nn
from torch.utils.data import random_split,DataLoader,Dataset

class BilingualDataLoader(Dataset):
    def __init__(self,ds,source_tokenizer,target_tokenizer,source_lang,target_lang,seq_len) -> None:
        super().__init__()
        
        self.seq_len = seq_len
        self.ds = ds
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.sos_token = torch.tensor([target_tokenizer.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token = torch.tensor([target_tokenizer.token_to_id("[EOS]")],dtype=torch.int64)
        self.pad_token = torch.tensor([target_tokenizer.token_to_id("[PAD]")],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        source_target_pair= self.ds[index]
        source_text=source_target_pair['translation'][self.source_lang]
        target_text=source_target_pair['translation'][self.target_lang]

        encoder_input_tokens=self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens=self.target_tokenizer.encode(target_text).ids

        encoder_pad_len = self.seq_len - len(encoder_input_tokens) - 2
        decoder_pad_len = self.seq_len - len(decoder_input_tokens) - 1 

        if encoder_pad_len < 0  or decoder_pad_len < 0:
            raise ValueError("Sentence is too long")
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_pad_len, dtype=torch.int64),
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_pad_len, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_pad_len, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": source_text,
            "tgt_text": target_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
    
        
