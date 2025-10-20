import torch
import argparse
import os
import math
from collections import Counter
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# -----------------------------
# Utilities: vocab, tokenizer, positional encoding
# -----------------------------
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
    def __call__(self, text):
        tokens = ['<bos>'] + text.strip().split() + ['<eos>']
        return [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]

def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for s in sentences:
        counter.update(s.strip().split())
    vocab = {word: idx+4 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab.update({'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3})
    return vocab

def load_or_build_vocab(file_path, cache_path, min_freq=1):
    if os.path.exists(cache_path):
        print(f"Loading cached vocab from {cache_path}")
        vocab = torch.load(cache_path)
    else:
        print(f"Building vocab from {file_path}")
        with open(file_path, encoding="utf-8") as f:
            sentences = f.readlines()
        vocab = build_vocab(sentences, min_freq=min_freq)
        torch.save(vocab, cache_path)
        print(f"Saved vocab to {cache_path}")
    return vocab

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
        return self.dropout(x + self.pe[:x.size(0)])

# -----------------------------
# Transformer model
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
    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)
    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

# -----------------------------
# Main inference function
# -----------------------------
def translate(model, sentence, src_tokenizer, tgt_vocab, device):
    model.eval()
    src = torch.tensor(src_tokenizer(sentence), dtype=torch.long).unsqueeze(1).to(device)
    src_mask = torch.zeros((src.size(0), src.size(0)), device=device)
    memory = model.encode(src, src_mask)
    tgt_indices = [tgt_vocab['<bos>']]

    for _ in range(50):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(1).to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0)).to(device)
        out = model.decode(tgt_tensor, memory, tgt_mask)
        out = model.generator(out)
        next_token = out.argmax(-1)[-1, 0].item()
        tgt_indices.append(next_token)
        if next_token == tgt_vocab['<eos>']:
            break

    inv_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}
    return " ".join([inv_tgt_vocab.get(idx, "<unk>") for idx in tgt_indices[1:-1]])

# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for trained Seq2Seq Transformer")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--src_file", type=str, default="data/en.txt", help="File to rebuild src vocab")
    parser.add_argument("--tgt_file", type=str, default="data/it.txt", help="File to rebuild tgt vocab")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory for cached vocabs")
    parser.add_argument("--sentence", type=str, required=True, help="Sentence to translate")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    device = torch.device("cpu")

    # Load or build vocabs
    src_vocab = load_or_build_vocab(args.src_file, os.path.join(args.cache_dir, "src_vocab.pt"))
    tgt_vocab = load_or_build_vocab(args.tgt_file, os.path.join(args.cache_dir, "tgt_vocab.pt"))
    src_tokenizer = Tokenizer(src_vocab)

    # Load model
    model = Seq2SeqTransformer(3, 3, 256, len(src_vocab), len(tgt_vocab)).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Translate
    translation = translate(model, args.sentence, src_tokenizer, tgt_vocab, device)
    print(f'\nInput: {args.sentence}') 
    print(f'Translation: {translation}')
