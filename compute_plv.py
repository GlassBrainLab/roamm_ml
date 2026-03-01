import os
import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_time

# ==========================
# Config
# ==========================
data_root = "/gpfs1/pi/djangraw/mindless_reading/data"
sfreq = 256
window_seconds = 2
window_size = int(sfreq * window_seconds)

# ==========================
# EEG band definitions (global)
# ==========================
band_names = [
    "theta", "alpha", "beta", "gamma"
]
band_defs = [
    (4.0, 8.0),    # theta
    (8.5, 13.0),  # alpha
    (13.5, 30.0),  # beta
    (30.5, 49.5),  # gamma
]
n_bands = len(band_defs)

# For PLV connectivity, we need to define seed-target pairs. Here we will compute connectivity 
# for all pairs of the 64 channels. This will give us 64*63/2 = 2016 features per band before flattening.
n_channels = 64
rows, cols = np.tril_indices(n_channels, k=-1)
indices = (np.array(rows), np.array(cols))
pair_names = None

# ==========================
# Collect subjects
# ==========================
all_subjects = sorted(
    d for d in os.listdir(data_root)
    if d.startswith("s") and os.path.isdir(os.path.join(data_root, d))
)

# ==========================
# Main loop over subjects
# ==========================
for subject_id in all_subjects:
    # if subject_id not in ['s10103', 's10185']:
    #     continue  # for testing only one subject
    # define subject directory and file paths
    print(f"Processing subject: {subject_id}")
    subject_dir = os.path.join(
        data_root,
        subject_id,
        "ml_data",
        f"{window_size}window_datasets",
    )
    data_file = os.path.join(subject_dir, f"{subject_id}_{window_size}windowed_data.npy")
    label_file = os.path.join(subject_dir, f"{subject_id}_{window_size}windowed_labels.npy")
    col_file = os.path.join(subject_dir, f"{subject_id}_col_names.npy")
    # Skip subjects with missing files
    if not (os.path.exists(data_file) and os.path.exists(label_file) and os.path.exists(col_file)):
        print(f"Missing files for {subject_id}, skipping.")
        epoch_counts.append(0)
        continue
    # Load data, labels, and column names
    data = np.load(data_file)
    labels = np.load(label_file)
    col_name = np.load(col_file)

    # generate pair names if not provided
    if pair_names is None:
        pair_names = [f"{col_name[r]}-{col_name[c]}" for r, c in zip(rows, cols)]
    
    # get epoch count for this subject
    epoch_count = data.shape[0]
    # check sample count for every subject
    if epoch_count < 10:
        print(f"Subject {subject_id} has insufficient data samples: {epoch_count} samples.")
        continue

    # ==========================
    # EEG PLV features
    # ==========================
    # get the 64-channel EEG data (first 64 cols)
    eeg_data = data[:, :64, :].astype(np.float64)

    # define an empty array to hold PLV features: (n_epochs, n_pairs, n_bands)
    plv = np.zeros((epoch_count, len(rows), n_bands), dtype=np.float64)
    # loop over bands to compute PLV features for each band separately
    for f_idx, freqs in enumerate(band_defs):
        con = spectral_connectivity_time(
                eeg_data,
                freqs,
                method="plv",
                sfreq=sfreq,
                faverage=True,
                indices=indices,
                n_cycles=5,
                verbose=False,
            )
        plv[:, :, f_idx] = con.get_data().squeeze()

    # flatten PLV features to (n_epochs, n_pairs * n_bands) and create column names
    df_plv = pd.DataFrame(plv.reshape(epoch_count, -1), columns=[f"plv_{band}_{pair}" for band in band_names for pair in pair_names])
    # add labels and subject_id to the dataframe
    df_plv["label"] = labels
    df_plv["subject_id"] = subject_id
    # save the PLV features to CSV
    out_file = os.path.join(
        subject_dir,
        f"{subject_id}_{window_size}windowed_plv.csv",
    )
    df_plv.to_csv(out_file, index=False)
    print(f"Saved PLV features for {subject_id} to: {out_file}")
