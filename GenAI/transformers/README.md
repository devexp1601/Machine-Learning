# Train Text Generator (PyTorch)

This folder contains a small decoder-only Transformer that learns next-character prediction from a text file.

## 1) Install dependencies

From the workspace root:

```bash
pip install torch
```

## 2) Go to this folder

```bash
cd GenAI/transformers
```

## 3) Start training (basic)

```bash
python train_text_generator.py --data-path ../../data.txt
```

## 4) Start training (custom settings)

```bash
python train_text_generator.py --data-path ../../data.txt --max-iters 5000 --batch-size 32 --block-size 64 --save-path checkpoints/model.pt
```

## 5) What you will see

- Train/validation loss printed every `--eval-interval` steps.
- Saved model checkpoint at `--save-path` (default: `model.pt`).
- Sample generated text at the end.

## 6) Useful command

Show all options:

```bash
python train_text_generator.py --help
```

## Optional: run from workspace root instead of this folder

```bash
python GenAI/transformers/train_text_generator.py --data-path data.txt
```
