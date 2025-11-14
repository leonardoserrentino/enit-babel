# EnItBabel
A compact, didactic PyTorch implementation of a Seq2Seq Transformer for word-level machine translation.
It includes a full training pipeline, checkpointing, and a CLI for greedy-decoding inference.
Great for teaching, small experiments, and quick baselines. 

--------------------------------------------------
FEATURES
--------------------------------------------------
- Plain-English training script with logging, batching, masking, and checkpointing.
- Self-contained inference CLI.
- Minimal dependencies: PyTorch + standard Python.
- Word-level tokenization with ```<pad>/<unk>/<bos>/<eos>```.

--------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------
```@txt
train_model.py        -> Train the Seq2Seq Transformer and save checkpoints
inference.py          -> Load a checkpoint and translate a sentence
data/
  en.txt              -> Source sentences (one per line)
  it.txt              -> Target sentences (one per line)
checkpoints/          -> Auto-created during training
```
--------------------------------------------------
INSTALLATION
--------------------------------------------------
```@bash
python -m venv .venv
source .venv/bin/activate
pip install torch
```
--------------------------------------------------
DATA & VOCABULARY
--------------------------------------------------
Input: plain text (one sentence per line, UTF-8).
Vocabulary built dynamically with special tokens.
Tokenization: whitespace split with <bos>/<eos> around each sentence.

--------------------------------------------------
TRAINING
--------------------------------------------------
This config took 21 days to complete.
```@bash
python train_model.py
        --src ../data/en.txt
        --tgt ../data/it.txt
        --epochs 10
        --batch_size 32
        --threads 4
```
--------------------------------------------------
CHECKPOINTS & BY-PRODUCTS
--------------------------------------------------
Each epoch saves a checkpoint in ./checkpoints/ containing:
- epoch
- model_state_dict
- optimizer_state_dict
- loss

--------------------------------------------------
INFERENCE
--------------------------------------------------
```@bash
python inference.py
      --model checkpoints/checkpoint_epoch_10.pt
      --src_file data/en.txt
      --tgt_file data/it.txt
      --sentence "I like machine learning"
```
--------------------------------------------------
MODEL ARCHITECTURE
--------------------------------------------------
Embedding + PositionalEncoding -> nn.Transformer -> Linear generator

--------------------------------------------------
LIMITATIONS
--------------------------------------------------
- Word-level tokenization (no BPE/WordPiece)
- Greedy decoding only
- CPU-only (no GPU flag)

--------------------------------------------------
REPRODUCIBILITY
--------------------------------------------------
Random seeds not fixed; results may vary.

--------------------------------------------------
ROADMAP
--------------------------------------------------
- Add BPE subword tokenization
- Add beam search and BLEU evaluation
- GPU/AMP support

--------------------------------------------------
CONTACT
--------------------------------------------------
Need the original data? Contact me via email.

--------------------------------------------------
LICENSE
--------------------------------------------------
MIT License
