import math
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# -----------------------------
# Dataset
# -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer):
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_lines = [line.strip() for line in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_lines = [line.strip() for line in f]
        assert len(self.src_lines) == len(self.tgt_lines), "Mismatch lines"
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = torch.tensor(self.src_tokenizer(self.src_lines[idx]), dtype=torch.long)
        tgt_tokens = torch.tensor(self.tgt_tokenizer(self.tgt_lines[idx]), dtype=torch.long)
        return src_tokens, tgt_tokens


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    return src_batch, tgt_batch

# -----------------------------
# Vocab & Tokenizer class
# -----------------------------
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
    def __call__(self, text):
        tokens = ['<bos>'] + text.split() + ['<eos>']
        return [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]


def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    vocab = {word: idx+4 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab.update({'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3})
    return vocab

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# -----------------------------
# Seq2Seq Transformer
# -----------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, nhead=8):
        super().__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

# -----------------------------
# Masks
# -----------------------------
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

def create_mask(src, tgt, pad_idx=0):
    src_seq_len, tgt_seq_len = src.size(0), tgt.size(0)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type_as(tgt_mask)
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# -----------------------------
# Training
# -----------------------------
def train_epoch(model, optimizer, dataloader, loss_fn, pad_idx):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader, start=1):
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            logging.info(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Seq Transformer Training")
    parser.add_argument("--src", type=str, default="../data/en.txt", help="Source file path")
    parser.add_argument("--tgt", type=str, default="../data/it.txt", help="Target file path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for torch and DataLoader")
    args = parser.parse_args()

    # Threads
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    # Checkpoints directory
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.info(f"Checkpoint directory: {checkpoint_dir}")

    # Load sentences
    with open(args.src, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f]
    with open(args.tgt, encoding="utf-8") as f:
        tgt_lines = [line.strip() for line in f]

    # Build vocabulary and tokenizers
    src_vocab = build_vocab(src_lines)
    tgt_vocab = build_vocab(tgt_lines)
    src_tokenizer = Tokenizer(src_vocab)
    tgt_tokenizer = Tokenizer(tgt_vocab)

    # Dataset, DataLoader
    dataset = TranslationDataset(args.src, args.tgt, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn,
                            shuffle=True,
                            num_workers=args.threads)

    # Model, Optimizer, Loss
    model = Seq2SeqTransformer(num_encoder_layers=3,
                                num_decoder_layers=3,
                                emb_size=256,
                                src_vocab_size=len(src_vocab),
                                tgt_vocab_size=len(tgt_vocab))
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # Training
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}/{args.epochs}")
        epoch_loss = train_epoch(model, optimizer, dataloader, loss_fn, pad_idx=0)
        logging.info(f"Epoch {epoch} completed, Avg Loss: {epoch_loss:.4f}")
        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, ckpt_path)
        logging.info(f"Checkpoint saved: {ckpt_path}")
