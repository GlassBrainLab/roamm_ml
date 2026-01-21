import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ROAMMEEG2TextDataset(Dataset):
    """CSV-based EEG→Text dataset.

    Expected columns (based on your upload):
      - 512 EEG bandpower columns (64 channels × 8 bands), numeric
      - sentence_id: groups timepoints into a sentence
      - sentence: the target text for that sentence
      - fix_R_fixed_word_key: token to identify fixation/word being fixated at each timepoint

    This dataset creates *fixation-event tokens* by collapsing consecutive identical
    fix_R_fixed_word_key values into one event, and averaging EEG features within the event.

    Returns the same tuple structure as the original GitHub ZuCo Dataset:
      (input_embeddings, seq_len, input_masks, input_mask_invert,
       target_ids, target_mask, sentiment_labels, sent_level_EEG)
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        tokenizer,
        seed: int = 42,
        max_seq_len: int = 64,
        max_target_len: int = 64,
        sentence_id_col: str = "sentence_id",
        sentence_col: str = "sentence",
        fix_key_col: str = "fix_R_fixed_word_key",
        data_type: str = "all",
    ):
        assert split in {"train", "dev", "test"}, "split must be one of: train, dev, test"
        self.csv_path = csv_path
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.max_target_len = int(max_target_len)
        self.sentence_id_col = sentence_id_col
        self.sentence_col = sentence_col
        self.fix_key_col = fix_key_col

        df = pd.read_csv(csv_path)

        # Filter by reading state
        # data_type:
        #   - "all": keep all
        #   - "nr":  keep only normal reading (is_mw == 0)
        #   - "mw":  keep only mind-wandering (is_mw != 0)
        assert data_type in {"all", "nr", "mw"}, f"Invalid data_type={data_type}"

        if data_type != "all":
            if "is_mw" not in df.columns:
                raise ValueError("Column 'is_mw' not found in CSV but data_type != 'all' was requested.")
            if data_type == "nr":
                df = df[df["is_mw"] == 0].reset_index(drop=True)
            elif data_type == "mw":
                df = df[df["is_mw"] != 0].reset_index(drop=True)

        # Identify EEG feature columns: numeric columns excluding obvious metadata/labels.
        meta_cols = {
            sentence_id_col, sentence_col, fix_key_col,
            "fix_R_fixed_word", "is_mw", "page_num", "story_name", "subject_id",
        }
        eeg_cols = [c for c in df.columns if c not in meta_cols]
        eeg_cols = [c for c in eeg_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(eeg_cols) == 0:
            raise ValueError("No numeric EEG feature columns found after excluding metadata columns.")
        self.eeg_cols = eeg_cols

        # Drop whole sentences containing any NaN in EEG features or missing sentence text
        # if drop_nan_sentences:
        #     good_ids = []
        #     for sid, g in df.groupby(sentence_id_col, sort=False):
        #         if g[self.sentence_col].isna().any():
        #             continue
        #         if g[self.eeg_cols].isna().any().any():
        #             continue
        #         good_ids.append(sid)
        #     df = df[df[sentence_id_col].isin(good_ids)].reset_index(drop=True)

        # Split by unique sentence text (unseen sentences in test)
        sent_ids = df[sentence_col].dropna().unique().tolist()
        rng = np.random.RandomState(seed)
        rng.shuffle(sent_ids)

        n = len(sent_ids)
        n_train = int(round(0.8 * n))
        n_dev = int(round(0.1 * n))

        train_ids = set(sent_ids[:n_train])
        dev_ids   = set(sent_ids[n_train:n_train + n_dev])
        test_ids  = set(sent_ids[n_train + n_dev:])

        split_ids = {"train": train_ids, "dev": dev_ids, "test": test_ids}[split]
        self.df = df[df[sentence_col].isin(split_ids)].reset_index(drop=True)

        # Store grouped sentence views for indexing
        self.sent_groups = list(self.df.groupby(sentence_col, sort=False))

    def __len__(self):
        return len(self.sent_groups)

    @staticmethod
    def _make_fix_event_ids(fix_keys: np.ndarray) -> np.ndarray:
        """Create fixation-event ids by incrementing when fix_key changes."""
        if len(fix_keys) == 0:
            return np.array([], dtype=np.int64)

        fk = []
        for x in fix_keys:
            if x is None:
                fk.append("<NA>")
            elif isinstance(x, float) and np.isnan(x):
                fk.append("<NA>")
            else:
                fk.append(str(x))

        fk = np.asarray(fk, dtype=object)
        change = np.ones(len(fk), dtype=np.int64)
        change[1:] = (fk[1:] != fk[:-1]).astype(np.int64)
        return np.cumsum(change) - 1  # start from 0

    def __getitem__(self, idx):
        sid, g = self.sent_groups[idx]

        # Target text (full sentence)
        target_text = str(g[self.sentence_col].iloc[0])

        # Build fixation-event tokens from timepoints
        fix_keys = g[self.fix_key_col].to_numpy()
        event_id = self._make_fix_event_ids(fix_keys)

        g2 = g.copy()
        g2["_event_id"] = event_id

        feats = (
            g2.groupby("_event_id")[self.eeg_cols]
              .mean()
              .to_numpy(dtype=np.float32)
        )  # [seq_len, n_feat]

        seq_len = feats.shape[0]
        if seq_len >= self.max_seq_len:
            feats = feats[: self.max_seq_len]
            seq_len_eff = self.max_seq_len
        else:
            seq_len_eff = seq_len

        # Pad to fixed max_seq_len (so default PyTorch collate works)
        input_embeddings = np.zeros((self.max_seq_len, len(self.eeg_cols)), dtype=np.float32)
        input_embeddings[:seq_len_eff] = feats[:seq_len_eff]

        # Masks
        input_masks = np.zeros((self.max_seq_len,), dtype=np.int64)
        input_masks[:seq_len_eff] = 1

        # src_key_padding_mask: True where padding
        input_mask_invert = (input_masks == 0)  # boolean numpy array

        tok = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_ids = tok["input_ids"].squeeze(0).to(torch.long)
        target_mask = tok["attention_mask"].squeeze(0).to(torch.long)

        # Placeholders (kept for API compatibility with original code)
        sentiment_labels = torch.tensor(0, dtype=torch.long)
        sent_level_EEG = torch.tensor(0.0, dtype=torch.float32)

        return (
            torch.from_numpy(input_embeddings),                  # [max_seq_len, n_feat]
            torch.tensor(seq_len_eff, dtype=torch.long),         # scalar
            torch.from_numpy(input_masks).to(torch.long),      # HF attention_mask likes long
            torch.from_numpy(input_mask_invert).to(torch.bool),# <-- IMPORTANT
            target_ids,                                          # [max_target_len]
            target_mask,                                         # [max_target_len]
            sentiment_labels,                                    # scalar
            sent_level_EEG,                                      # scalar
        )
