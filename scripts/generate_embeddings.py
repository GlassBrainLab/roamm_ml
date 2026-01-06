#!/usr/bin/env python3
"""Generate static embeddings (GloVe + Word2Vec) for story CSVs in `res/`.

Usage:
  python scripts/generate_embeddings.py --res-dir res --out-dir res/embeddings

This script will:
 - detect CSV files in `--res-dir`
 - choose the text column automatically (longest average string)
 - load pretrained GloVe from gensim.downloader ('glove-wiki-gigaword-100')
 - try to load pretrained Word2Vec ('word2vec-google-news-300'); if unavailable,
   it will train a Word2Vec model on the available stories
 - write per-story embeddings and vocabs under `--out-dir/<story_basename>/`
"""
import argparse
import os
import re
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import numpy as np
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import gensim.downloader as api
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, KeyedVectors

lemmatizer = WordNetLemmatizer()

def clean_word(w):
    w = w.lower()
    w = re.sub(r"[^a-z0-9]", "", w)  # keep only a–z and 0–9
    if w:  # avoid empty strings
        w = lemmatizer.lemmatize(w, pos="n")  # plural → singular (nouns)
    return w

def load_stories(csv_paths):
    stories = {}
    for p in csv_paths:
        df = pd.read_csv(p)

        words = (
            df["words"]
            .dropna()
            .astype(str)
            .apply(clean_word)
        )

        # remove empty results after cleaning
        words = words[words != ""]

        stories[p.stem] = " ".join(words.tolist())

    return stories

def tokenize(text):
    return simple_preprocess(text, deacc=True)


def save_embeddings(out_dir: Path, name: str, vocab, vectors: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    # save as npy and vocab csv for readability
    np.save(out_dir / f"{name}_vectors.npy", vectors)
    pd.DataFrame({"token": vocab}).to_csv(out_dir / f"{name}_vocab.csv", index_label="index")


def vector_to_list(vec):
    if vec is None:
        return None
    return [float(x) for x in vec]


def process_csv_write_embeddings(csv_path: Path, glove, w2v, contextual_model=None, window_size=5):
    df = pd.read_csv(csv_path)
    if "words" not in df.columns:
        print(f"CSV {csv_path.name} has no 'words' column; skipping per-row embedding write.")
        return

    glove_col = []
    w2v_col = []
    contextual_col = []

    words_list = df["words"].astype(str).tolist()

    # Build contexts for contextual embeddings if model provided
    contexts = []
    if contextual_model is not None:
        n = len(words_list)
        for i in range(n):
            start = max(0, i - window_size)
            end = min(n, i + window_size + 1)
            ctx = " ".join(words_list[start:end])
            contexts.append(ctx)
        # encode all contexts in batch depending on backend
        if isinstance(contextual_model, tuple) and contextual_model[0] == "gpt2":
            # contextual_model = ("gpt2", model, tokenizer, device)
            _, gpt2_model, gpt2_tokenizer, gpt2_device = contextual_model
            ctx_vecs = gpt2_encode(contexts, gpt2_model, gpt2_tokenizer, gpt2_device)
        else:
            ctx_vecs = contextual_model.encode(contexts, show_progress_bar=True, batch_size=32)

    for idx, w in enumerate(words_list):
        w_lower = w.strip()
        # GloVe
        try:
            gvec = glove.get_vector(w_lower) if hasattr(glove, "get_vector") else glove[w_lower]
            glove_col.append(json.dumps(vector_to_list(gvec)))
        except Exception:
            glove_col.append("")

        # Word2Vec
        try:
            wvec = w2v.get_vector(w_lower) if hasattr(w2v, "get_vector") else w2v[w_lower]
            w2v_col.append(json.dumps(vector_to_list(wvec)))
        except Exception:
            w2v_col.append("")

        # Contextual
        if contextual_model is not None:
            vec = ctx_vecs[idx]
            contextual_col.append(json.dumps(vector_to_list(vec)))
        else:
            contextual_col.append("")

    df["glove"] = glove_col
    df["word2vec"] = w2v_col
    df["contextual"] = contextual_col
    # overwrite the CSV with new columns
    df.to_csv(csv_path, index=False)


def gpt2_encode(texts, model, tokenizer, device, batch_size=16):
    all_vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs.last_hidden_state shape: (batch, seq_len, hidden)
            last = outputs.last_hidden_state
            # mean pool over non-padded tokens
            mask = attention_mask.unsqueeze(-1).expand(last.size()).float()
            summed = torch.sum(last * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            all_vecs.append(mean_pooled)
    return np.vstack(all_vecs)


def main(res_dir: Path, out_dir: Path, glove_name: str = "glove-wiki-gigaword-100"):
    csv_paths = sorted(res_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSVs found in {res_dir}")

    print(f"Found {len(csv_paths)} CSV files; using: {[p.name for p in csv_paths]}")

    stories = load_stories(csv_paths)
    tokenized = {k: tokenize(v) for k, v in stories.items()}

    print("Loading GloVe model (this will download if needed):", glove_name)
    glove = api.load(glove_name)

    # Try loading pretrained Word2Vec
    w2v = None
    try:
        print("Attempting to load pretrained Word2Vec (word2vec-google-news-300)")
        w2v = api.load("word2vec-google-news-300")
    except Exception as e:
        print("Pretrained Word2Vec not available or failed to load; will train on corpus.")

    if w2v is None:
        # train Word2Vec on all tokenized stories
        print("Training Word2Vec (this may take a moment)...")
        sentences = list(tokenized.values())
        model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=1, epochs=10)
        w2v = model.wv

    # For each story, build vocab intersection and save; also write per-row embeddings into CSVs
    for name, tokens in tqdm(tokenized.items(), desc="Stories"):
        unique_tokens = sorted(set(tokens))
        glove_vecs = []
        glove_vocab = []
        for t in unique_tokens:
            try:
                vec = glove.get_vector(t) if hasattr(glove, "get_vector") else glove[t]
                glove_vecs.append(vec)
                glove_vocab.append(t)
            except KeyError:
                continue

        w2v_vecs = []
        w2v_vocab = []
        for t in unique_tokens:
            try:
                vec = w2v.get_vector(t) if hasattr(w2v, "get_vector") else w2v[t]
                w2v_vecs.append(vec)
                w2v_vocab.append(t)
            except KeyError:
                continue

        save_embeddings(out_dir / name, "glove", glove_vocab, np.vstack(glove_vecs) if glove_vecs else np.empty((0, glove.vector_size)))
        save_embeddings(out_dir / name, "word2vec", w2v_vocab, np.vstack(w2v_vecs) if w2v_vecs else np.empty((0, w2v.vector_size)))

    # Load contextual model: prefer GPT-2 for contextualized embeddings, fallback to SBERT
    model_contextual = None
    # Try GPT-2 backend
    try:
        print("Loading GPT-2 model/tokenizer for contextual embeddings (gpt2)")
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT-2 has no pad token by default; set pad_token to eos_token for batching
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = AutoModel.from_pretrained("gpt2")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt2_model.to(device)
        model_contextual = ("gpt2", gpt2_model, gpt2_tokenizer, device)
    except Exception:
        print("Could not load GPT-2; falling back to sentence-transformers model.")
        try:
            print("Loading contextual sentence-transformers model (all-MiniLM-L6-v2)")
            model_contextual = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            print("Could not load sentence-transformers model; contextual embeddings will be empty.")

    # Now update each CSV in res_dir with per-row embeddings if 'words' column exists
    for p in csv_paths:
        print(f"Writing per-row embeddings into {p.name}")
        process_csv_write_embeddings(p, glove, w2v, contextual_model=model_contextual, window_size=5)

    print("Done. Embeddings saved under:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", type=Path, default=Path("res"), help="Directory containing story CSVs")
    parser.add_argument("--out-dir", type=Path, default=Path("res/embeddings"), help="Output directory")
    args = parser.parse_args()
    main(args.res_dir, args.out_dir)
