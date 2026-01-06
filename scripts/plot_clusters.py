#!/usr/bin/env python3
"""Plot clustering results for nouns and verbs across embedding types.

Produces t-SNE visualizations for clusters found by KMeans, Agglomerative, and DBSCAN
for each embedding type: `glove`, `word2vec`, and `contextual`.

Saves PNGs to `res/embeddings/clusters_<emb>_<method>.png` and combined figures.
"""
from pathlib import Path
import json
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
import nltk
import spacy
nlp = spacy.load("en_core_web_sm")

def keep_nouns_verbs(df, word_col="words"):
    docs = nlp.pipe(df[word_col].astype(str), batch_size=1000)

    pos = []
    for doc in docs:
        # each doc should contain exactly one token
        token = doc[0]
        pos.append(token.pos_)

    df = df.copy()
    df["pos"] = pos

    return df[df["pos"].isin(["NOUN", "VERB"])]

def normalize_token(t: str):
    t = str(t).strip()
    # lower and strip surrounding punctuation
    t = t.lower()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t

def cluster_and_plot(tokens, mat, methods, out_prefix: Path, emb_name: str):
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    coords = tsne.fit_transform(mat)
    for name, model in methods.items():
        try:
            labels = model.fit_predict(mat)
        except Exception:
            labels = model.fit_predict(mat)
        fig, ax = plt.subplots(figsize=(10, 8))
        unique = np.unique(labels)

        for lab in unique:
            mask = labels == lab

            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                label=str(lab),
                alpha=0.85,
                edgecolors="w",
                linewidths=0.3,
            )

        # annotate every point with its token label
        for i, t in enumerate(tokens):
            # small offset so text does not sit exactly on the marker
            ax.annotate(
                t,
                xy=(coords[i, 0], coords[i, 1]),
                xytext=(2, 2),
                textcoords='offset points',
                fontsize=6,
                ha='left',
                va='bottom',
                clip_on=True,
            )
        # emb_name may include story prefix already (cleaned)
        ax.set_title(f'{emb_name} clusters ({name}) — n={len(tokens)}')
        ax.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        out = out_prefix / f'clusters_{emb_name}_{name}.png'
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print('Saved', out)


def main():
    base = Path('res')
    emb_dir = base / 'embeddings'
    csvs = sorted(base.glob('*.csv'))

    methods = {
        'kmeans': KMeans(n_clusters=8, random_state=42),
        'agg': AgglomerativeClustering(n_clusters=8)
    }

    for csv in csvs:
        df = pd.read_csv(csv)
        # choose token column per-file
        word_col = 'word' if 'word' in df.columns else 'words'
        df_nv = keep_nouns_verbs(df, word_col=word_col)

        # clean story name (remove trailing '_control_coordinates') for titles and folders
        clean_story = csv.stem.replace('_control_coordinates', '')
        story_out = emb_dir / clean_story
        story_out.mkdir(parents=True, exist_ok=True)

        print(f'Processing story {clean_story} ({csv.name}): {len(df_nv)} noun/verb rows')

        for emb_name in ['glove', 'word2vec']:
            toks = []
            mats = []
            for _, row in df_nv.iterrows():
                val = row.get(emb_name, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                # parse JSON array if stored as string
                try:
                    if isinstance(val, str):
                        arr = json.loads(val)
                    else:
                        arr = val
                    vec = np.array(arr, dtype=float)
                except Exception:
                    continue
                token_raw = str(row.get(word_col, ''))
                token = normalize_token(token_raw)
                if not token:
                    continue
                toks.append(token)
                mats.append(vec)

            if not toks:
                print(f'No tokens for {emb_name} in {csv.name}')
                continue

            mat = np.vstack(mats)
            # prefix output with story name for clarity
            out_prefix = story_out
            cluster_and_plot(toks, mat, methods, out_prefix, f'{clean_story}_{emb_name}')


if __name__ == '__main__':
    main()
