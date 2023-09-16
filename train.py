from pathlib import Path
from config import get_config, get_weights_file_path
from dataset import BilingualDataLoader
from model import build_transformer

import torch
import torch.nn as nn
from torch.utils.data import random_split,DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer



def get_all_sentences(ds,lang):
     for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    print(f'Lang: {lang}')
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


def get_ds(config):
    ds_raw=load_dataset('opus_books',config['src_lang']+'-'+config['target_lang'],split='train')
    tokenizer_source = get_or_build_tokenizer(config,ds_raw,config['src_lang'])
    tokenizer_target = get_or_build_tokenizer(config,ds_raw,config['target_lang'])

    train_ds_size= int(len(ds_raw)*0.9)
    val_ds_size = len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataLoader(train_ds_raw, tokenizer_source, tokenizer_target, config['src_lang'], config['target_lang'], config['seq_len'])
    val_ds = BilingualDataLoader(val_ds_raw, tokenizer_source, tokenizer_target, config['src_lang'], config['target_lang'], config['seq_len'])

    max_len_source = 0
    max_len_target = 0

    for item in ds_raw:
        source_ids = tokenizer_source.encode(item['translation'][config['src_lang']]).ids
        target_ids = tokenizer_target.encode(item['translation'][config['target_lang']]).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f'Max length of source sentence: {max_len_source}')
    print(f'Max length of target sentence: {max_len_target}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_source, tokenizer_target 



def get_model(config, vocab_source_len,vocab_target_len):
    model=build_transformer(vocab_source_len,vocab_target_len,config['seq_len'],config['seq_len'],config['d_model'])
    return model


def train_model(config):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_source, tokenizer_target = get_ds(config)
    model= get_model(config, tokenizer_source.get_vocab_size(),tokenizer_target.get_vocab_size()).to(device)
    writer= SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_source.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
            label = batch['label'].to(device) # (B, seq_len)

            loss = loss_fn(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
