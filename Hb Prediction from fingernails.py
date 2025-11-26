import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as skio
import scipy.stats as stats
from xgboost import XGBRegressor

import sklearn
from sklearn.model_selection import GridSearchCV, cross_val_predict, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.feature_selection import SelectKBest, f_regression 

import skimage.transform as sktf 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
import tensorflow as tf 
from keras.applications import DenseNet169 
from keras.applications.densenet import preprocess_input as densenet_preprocess

from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore', category=sklearn.exceptions.ConvergenceWarning)


# Configurations --------------------------------------------------------------------------------------------
metadata = pd.read_csv(r"C:\Users\julia\OneDrive - Tilburg University\Thesis\Benchmark\data\metadata.csv")
image_folder = r"C:\Users\julia\OneDrive - Tilburg University\Thesis\benchmark\data"
results_folder = r"C:\Users\julia\OneDrive - Tilburg University\Thesis\benchmark\Results"

sanquin_hb_csv = r"C:\Users\julia\OneDrive - Tilburg University\Thesis\Sanquin\data\hb_values.csv"
sanquin_image_folder = r"C:\Users\julia\OneDrive - Tilburg University\Thesis\Sanquin\data"
sanquin_results_folder = r"C:\Users\julia\OneDrive - Tilburg University\Thesis\Sanquin\Results"

seed        = 42
n_folds     = 7
n_nails     = 3

models_run  = [ "Elastic Net", "Random Forest", "XGBoost", "CNN + RF hybrid"]
eda_run     = True
error_run   = True
sanquin_run = True
                            
# EDA -------------------------------------------------------------------------------------------------------------
def run_eda(features_per_nail, y, results_folder):
    nail_idx = 0 if len(features_per_nail) == 1 else 1
    nail_df = features_per_nail[nail_idx].copy()
    nail_df = nail_df.assign(Hb=y.values)
    
    # Descriptive statistics
    print(f"Total samples (patients): {len(metadata)}")
    print("\n Hb summary statistics:\n", y.describe())

    # Hb distribution histogram
    plt.figure(figsize=(6,4))
    counts, bins, patches = plt.hist(
        nail_df["Hb"], bins=20, color="#4C72B0", alpha=0.7
    )
    plt.xlabel("Haemoglobin (g/L)")
    plt.ylabel("Count")
    plt.title("Distribution of Haemoglobin levels")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "EDA_hb_distribution.png"), dpi=150)
    plt.close()

    # Hb boxplot
    plt.figure(figsize=(5,3))
    plt.boxplot(nail_df["Hb"], vert=False, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", color="#4C72B0"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="#4C72B0"),
                capprops=dict(color="#4C72B0"),
                flierprops=dict(markerfacecolor="#4C72B0", marker='o', markersize=4, alpha=0.5))
    plt.title("Hb value range")
    plt.ylabel("Haemoglobin (g/L)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "EDA_hb_boxplot.png"), dpi=150)
    plt.close()

    # Range comparison 
    bins = [-np.inf, 120, 150, np.inf]
    labels = ["<120 (anaemic)", "120–150 (normal)", ">150 (high)"]
    nail_df["Hb_range"] = pd.cut(nail_df["Hb"], bins=bins, labels=labels, right=False)

    counts = nail_df["Hb_range"].value_counts().reindex(labels, fill_value=0)

    plt.figure(figsize=(6,4))
    colors = ["#E24A33", "#59A14F", "#4C72B0"]
    plt.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="black", alpha=0.8)
    plt.title("Count per Hb group")
    plt.xlabel("Hb Range (g/L)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "EDA_hb_group_counts.png"), dpi=150)
    plt.close()

def plot_rgb_percentiles_benchmark(features_per_nail, results_folder):
    bench = features_per_nail[0]

    # Histogram of mean intesnity per feature per channel
    channels = ["R", "G", "B"]
    colors = {
        "R": "#E24A33",
        "G": "#59A14F",
        "B": "#4C72B0" 
    }
    plt.figure(figsize=(13, 4))

    for i, chan in enumerate(channels, start=1):

        cols = [c for c in bench.columns if f"NAIL_{chan}_p=" in c]
        cols = sorted(cols, key=lambda s: int(s.split("=")[1]))

        x = np.arange(len(cols))
        mean_vals = [bench[c].mean() for c in cols]

        plt.subplot(1, 3, i)
        plt.bar(x, mean_vals, color=colors[chan])
        plt.xticks(x, [c.split("=")[1] for c in cols])
        plt.xlabel("Percentile")
        plt.title(f"{chan} channel")
        plt.ylim(0, 1.05) 
        if i == 1:
            plt.ylabel("Mean normalised value")

    plt.suptitle("RGB percentile features (Benchmark)")
    plt.tight_layout()
    out_path = os.path.join(results_folder, "EDA_rgb_percentiles_benchmark.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


# EDA for Sanquin dataset -------------------------------------------------------------------------
def run_eda_sanquin(hb_csv_path, image_dir, sanquin_results_folder):
    # Load Hb values
    df = pd.read_csv(hb_csv_path, sep=';')

    # Average Hb values for each ID and convert mmol/dL to g/L
    hb_cols = [c for c in ["Hb_1", "Hb_2", "Hb_3"] if c in df.columns]
    df["HB_LEVEL_GperL"] = df[hb_cols].mean(axis=1, skipna=True) * 1.61134386078 

    # Remove ID's without Hb 
    df = df[df["HB_LEVEL_GperL"].notna()].copy()

    # Counting available photos per ID
    def count_nails_for_id(pid_str):
        count = 0
        for k in range(1, 5):
            filename = f"ID_{pid_str}_NAIL_{k}.png"
            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                count += 1
        return count

    df["PATIENT_ID"] = df["ID"].apply(lambda x: f"{int(x):03d}")
    df["n_nails_available"] = df["PATIENT_ID"].apply(count_nails_for_id)

    # Exclude people without nails, if they exist
    df = df[df["n_nails_available"] > 0].copy()

    # EDA dataframe
    y = df["HB_LEVEL_GperL"]

    print(f"Total samples (patients with Hb and at least one nail): {len(df)}")
    print("\nHb summary statistics:\n", y.describe())

    # Hb distribution histogram
    plt.figure(figsize=(6,4))
    plt.hist(y.values, bins=20, color="#4C72B0", alpha=0.7)
    plt.xlabel("Haemoglobin (g/L)")
    plt.ylabel("Count")
    plt.title("Sanquin: Distribution of Haemoglobin levels")
    plt.tight_layout()
    plt.savefig(os.path.join(sanquin_results_folder, "EDA_sanquin_hb_distribution.png"), dpi=150)
    plt.close()

    # Hb boxplot
    plt.figure(figsize=(5,3))
    plt.boxplot(y.values, vert=False, patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", color="#4C72B0"),
                medianprops=dict(color="black"),
                whiskerprops=dict(color="#4C72B0"),
                capprops=dict(color="#4C72B0"),
                flierprops=dict(markerfacecolor="#4C72B0", marker='o', markersize=4, alpha=0.5))
    plt.title("Sanquin: Hb value range")
    plt.ylabel("Haemoglobin (g/L)")
    plt.tight_layout()
    plt.savefig(os.path.join(sanquin_results_folder, "EDA_sanquin_hb_boxplot.png"), dpi=150)
    plt.close()

    # Barplot of amount of nails available
    nail_bins = [1, 2, 3, 4]
    cumulative_counts = [
        (df["n_nails_available"] >= k).sum() for k in nail_bins
    ]

    plt.figure(figsize=(6,4))
    plt.bar([str(k) for k in nail_bins], cumulative_counts,
            color="#4C72B0", edgecolor="black", alpha=0.8)
    plt.title("Sanquin: Participants by minimum number of nails available")
    plt.xlabel("Minimum number of nails available")
    plt.ylabel("Number of participants")
    plt.tight_layout()
    plt.savefig(os.path.join(sanquin_results_folder, "EDA_sanquin_nail_counts.png"), dpi=150)
    plt.close()

# Preprocessing ------------------------------------------------------------------------------------------------------
# Stores cropped and normalized images
cnn_images_per_nail = None

# Crops images to 60%
def cut_image(img, low=0.2, high=0.8):
    h, w = img.shape[:2]
    return img[int(low*h):int(high*h), int(low*w):int(high*w), :]

# Computes the percentiles of each channel on the remaining crops
def calculate_features(img, percentiles=(5,15,25,50,75,85,95), low=0.2, high=0.8):
    features = {}
    patch = cut_image(img, low=low, high=high)
    for chan_id, color in enumerate("RGB"):
        arr = patch[:, :, chan_id].ravel()
        for p in percentiles:
            features[f"{color}_p={p}"] = np.percentile(arr, p)
    return features

def preprocess_data(metadata, image_folder, *, n_nails=1): 
    # Creates Python lists of the bounding boxes
    metadata["NAIL_BOUNDING_BOXES"] = metadata["NAIL_BOUNDING_BOXES"].apply(json.loads)

    # Keeps track of what white reference belongs to what nails
    images = defaultdict(list)
    ids = defaultdict(list)
    white_ref = []
    white_ids = []

    # Crops the images of nails and white patches
    print("[preprocessing] Extracting crops of benchmark")
    for j, (_, row) in enumerate(metadata.iterrows()):
        img_path = os.path.join(image_folder, f"{row.PATIENT_ID}.jpg")
        if not os.path.exists(img_path):
            continue

        img = skio.imread(img_path)
        nails_bboxes = row.NAIL_BOUNDING_BOXES

        for finger_num, (top, left, bot, right) in enumerate(nails_bboxes, start=1):
            key = f"NAIL_{finger_num}"
            images[key].append(img[top:bot, left:right])
            ids[key].append(row.PATIENT_ID)  

        white_ref.append(img[350:400, 300:350]) 
        white_ids.append(row.PATIENT_ID) 

    # Dataframe with the white-patch median values for normalization
    white_df = pd.DataFrame({
        "PATIENT_ID": white_ids,
        "R": [np.median(p[:, :, 0].ravel()) for p in white_ref],
        "G": [np.median(p[:, :, 1].ravel()) for p in white_ref],
        "B": [np.median(p[:, :, 2].ravel()) for p in white_ref],
    })

    # Decides what nails to choose when predicting from multiple
    if n_nails == 1:
        nails_to_use = [2]
    elif n_nails == 2:
        nails_to_use = [1, 2]
    else:
        nails_to_use = [1, 2, 3]

    # Computes the RGB features from the percentiles
    per_nail = []
    for k in nails_to_use:
        key = f"NAIL_{k}"
        if key not in images or len(images[key]) == 0:
            raise RuntimeError(f"No crops found for {key}.")

        features = pd.DataFrame([calculate_features(im) for im in images[key]])
        features = features.rename(columns=lambda c: f"NAIL_{c}")
        features.insert(0, "PATIENT_ID", ids[key])

        merged = features.merge(white_df, on="PATIENT_ID", how="inner")
        for col in list(merged.columns):
            if col.startswith("NAIL_"):
                color = col.split("_")[1]
                merged[col] = merged[col] / merged[color]
        keep_cols = ["PATIENT_ID"] + [c for c in merged.columns if c.startswith("NAIL_")]
        merged = merged[keep_cols].copy()
        merged = merged.sort_values("PATIENT_ID").reset_index(drop=True)
        merged = merged.set_index("PATIENT_ID")
        per_nail.append(merged)

    
    # Preprocessing part for the CNN images-------
    global cnn_images_per_nail
    cnn_images_per_nail = []
    # Loops through all chosen nails
    for k in nails_to_use:
        key = f"NAIL_{k}"
        # Matching the nails with their white reference
        df_ids = pd.DataFrame({"PATIENT_ID": ids[key]}).merge(white_df, on="PATIENT_ID", how="inner")
        norm_imgs, id_list = [], []
        # Loops over nail crops and their white references
        for im, (_, roww) in zip(images[key], df_ids.iterrows()):
            # Normalization with white references
            im = im.astype(np.float32)
            r, g, b = float(roww["R"]), float(roww["G"]), float(roww["B"])
            im[:,:,0] = im[:,:,0] / max(r, 1e-6)
            im[:,:,1] = im[:,:,1] / max(g, 1e-6)
            im[:,:,2] = im[:,:,2] / max(b, 1e-6)
            # Adjusting the brightness
            med = np.nanmedian(im)      
            if np.isfinite(med) and med > 0: 
                im = im / med * 128.0            
            im = np.clip(im, 0, 255)   
            # Using the same image cut as for the non CNN models
            im_crop = cut_image(im)              
            im_cut = sktf.resize(im_crop, (224, 224), preserve_range=True, anti_aliasing=True).astype(np.float32) 
            # Storing all images and corresponding ID's
            norm_imgs.append(im_cut)        
            id_list.append(roww["PATIENT_ID"])
        s = pd.Series(norm_imgs, index=pd.Index(id_list, name="PATIENT_ID"))
        s = s.groupby(level=0).first().sort_index()
        cnn_images_per_nail.append(s)  

    # Selects only patients that have the right amount of fingernails available
    common_ids = per_nail[0].index
    for d in per_nail[1:]:
        common_ids = common_ids.intersection(d.index)
    per_nail = [d.loc[common_ids] for d in per_nail]
    # Same but for CNN
    for i in range(len(cnn_images_per_nail)):
        cnn_images_per_nail[i] = cnn_images_per_nail[i].loc[common_ids]

    # Features put in list
    features_per_nail = per_nail

    # Makes a target vector of the patients
    sub = metadata.loc[metadata["PATIENT_ID"].isin(common_ids)].copy()
    sub = sub.sort_values("PATIENT_ID").reset_index(drop=True)
    y = sub["HB_LEVEL_GperL"]

    # Uses KDE sampling to create a "mask" of the train/test/validation set data
    np.random.seed(seed)
    kde_vals = stats.gaussian_kde(y, bw_method=0.5)(y)
    weights = (1 / kde_vals) / (1 / kde_vals).sum()
    patient_ids = sub["PATIENT_ID"]
    size = min(100, len(patient_ids))
    sampled_ids = np.random.choice(patient_ids, size=size, replace=False, p=weights)
    mask = sub["PATIENT_ID"].isin(sampled_ids).values

    # Density histogram before and after KDE balancing
    if eda_run:
        plt.figure(figsize=(4, 3), dpi=200)
        bins = np.linspace(float(sub["HB_LEVEL_GperL"].min()),float(sub["HB_LEVEL_GperL"].max()),16)

        plt.hist(
            sub["HB_LEVEL_GperL"],
            bins=bins,
            density=True,
            alpha=0.4,
            color="#4C72B0",
            linewidth=0.5,
            label="Before sampling"
        )

        plt.hist(
            sub.loc[mask, "HB_LEVEL_GperL"],
            bins=bins,
            density=True,
            alpha=0.4,
            color="#59A14F",
            linewidth=0.5,
            label="After sampling"
        )

        plt.xlabel("Hb level, g/L")
        plt.ylabel("Density")
        plt.legend(frameon=True, fontsize=8, loc="upper left")
        plt.tight_layout()
        out_overlay = os.path.join(results_folder, "EDA_kde_overlay.png")
        plt.savefig(out_overlay)
        plt.close()

    return features_per_nail, y, mask  

# Sanquin preprocessing -----------------------------------------------------------------------------------
def estimate_white_reference(img):

    img = img.astype(np.float32)

    r = np.percentile(img[:, :, 0], 99)
    g = np.percentile(img[:, :, 1], 99)
    b = np.percentile(img[:, :, 2], 99)

    return max(r, 1e-6), max(g, 1e-6), max(b, 1e-6)


def preprocess_sanquin(hb_csv_path, image_dir, n_nails=1):

    df = pd.read_csv(hb_csv_path, sep=';')

    # Average Hb values for each ID and convert mmol/dL to g/L
    hb_cols = [c for c in ["Hb_1", "Hb_2", "Hb_3"] if c in df.columns]
    df["HB_LEVEL_GperL"] = df[hb_cols].mean(axis=1, skipna=True) * 1.61134386078 

    # Remove ID's without Hb value
    df = df[df["HB_LEVEL_GperL"].notna()].copy()

    # Extract ID's
    df["PATIENT_ID"] = df["ID"].apply(lambda x: f"{int(x):03d}")

    # Creates list of dataframes for each nail, with ID as index
    features_per_nail = [[] for _ in range(n_nails)]
    ids = [[] for _ in range(n_nails)]

    for _, row in df.iterrows():
        pid = row["PATIENT_ID"]

        # Find all the available nails
        available = []
        for k in range(1, 5):
            filename = f"ID_{pid}_NAIL_{k}.png"
            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                available.append((k, img_path))

        # Skip ID's without enough nails
        if len(available) < n_nails:
            continue

        # Put the nails in order
        available.sort(key=lambda t: t[0])
        chosen = available[:n_nails]

        # Fill the list of dataframes with the nails
        for key, (_, img_path) in enumerate(chosen):
            img = skio.imread(img_path)

            # Use other version of white reference to match benchmark
            r_ref, g_ref, b_ref = estimate_white_reference(img)

            img_norm = img.astype(np.float32)     
            img_norm[:, :, 0] /= r_ref
            img_norm[:, :, 1] /= g_ref
            img_norm[:, :, 2] /= b_ref

            img_norm = np.clip(img_norm, 0, 255)  

            # Center crop of 50% to remove background and skin
            features = calculate_features(img_norm, low=0.25, high=0.75)
 
            # Store features and ID's
            features_per_nail[key].append(features)     
            ids[key].append(pid)     

    # Creating the dataframes per nail
    per_nail = []
    for key in range(n_nails):
        df_nail = pd.DataFrame(features_per_nail[key])
        df_nail = df_nail.rename(columns=lambda c: f"NAIL_{c}")
        df_nail.insert(0, "PATIENT_ID", ids[key])
        df_nail = df_nail.set_index("PATIENT_ID")
        per_nail.append(df_nail)

    # Keep only dataframes with enough nails
    common_ids = per_nail[0].index
    for d in per_nail[1:]:
        common_ids = common_ids.intersection(d.index)
    common_ids = common_ids.sort_values()

    # Amount of subjects included
    print(f"[Sanquin] Included {len(common_ids)} subjects with Hb and more than {n_nails} nails")

    # Matching Hb and features
    sub = df.set_index("PATIENT_ID").loc[common_ids].reset_index()
    y_sanquin = sub["HB_LEVEL_GperL"]

    per_nail = [d.loc[common_ids] for d in per_nail]

    return per_nail, y_sanquin

def plot_features_benchmark_vs_sanquin(
    benchmark_features_per_nail,
    sanquin_features_per_nail,
    nail_idx,
    out_dir,
    filename="EDA_rgb_p50_benchmark_vs_sanquin.png"
):
    channels = ["R", "G", "B"]
    plt.figure(figsize=(13, 4))

    for i, chan in enumerate(channels, start=1):
        feature = f"NAIL_{chan}_p=50"

        bench = benchmark_features_per_nail[nail_idx][feature].values
        sanq  = sanquin_features_per_nail[nail_idx][feature].values

        combined = np.concatenate([bench, sanq])
        bins = np.linspace(combined.min(), combined.max(), 40)

        plt.subplot(1, 3, i)
        plt.hist(bench, bins=bins, color='#4C72B0', density=True, alpha=0.5, label="Benchmark")
        plt.hist(sanq,  bins=bins, color='#59A14F', density=True, alpha=0.5, label="Sanquin")

        plt.title(f"Distribution of {chan}_p=50")
        plt.xlabel("Normalised value")
        if i == 1:
            plt.ylabel("Density")
            plt.legend()

    plt.suptitle("Distribution of channel percentile 50 in benchmark vs. Sanquin")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()


# CNN feature extraction (for hybrid)----------------------------------------------------------------------------
def feature_model():
    base = DenseNet169(include_top=False, weights="imagenet", input_shape=(224,224,3))
    # Building the feature extractor
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=x)
    model.trainable = False
    return model

def extract_cnn_features_per_nail(cnn_images_per_nail, batch_size=32):
    model = feature_model()
    results = []
    for s in cnn_images_per_nail:
        # Converts the list into an array for batch processing
        idx = s.index
        imgs = np.stack(s.values, axis=0).astype(np.float32)
        # Creating the features
        preprocessing = densenet_preprocess(imgs)
        features = model.predict(preprocessing, batch_size=batch_size, verbose=0)
        # L2 normalization for Random Forest
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms
        # Converts to data frame
        df = pd.DataFrame(features, index=idx)
        results.append(df)

    return results

# Tuning, Evaluation & Prediction-------------------------------------------------------------------------------------------
# Computes rmse and mae more cleanly
def compute_rmse_mae(y_true, y_pred):  
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  
    mae  = float(mean_absolute_error(y_true, y_pred)) 
    return rmse, mae

# Computes Bland-Altman plots
def plot_bland_altman(y_true, y_pred, title="Bland–Altman plot", savepath=None, show=False):
    diffs = y_pred - y_true
    means = (y_pred + y_true) / 2.0
    bias = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    loa_low, loa_high = bias - 1.96*sd, bias + 1.96*sd

    plt.figure(figsize=(7,5))
    plt.scatter(means, diffs, alpha=0.6, edgecolors='none')
    plt.axhline(bias, color='#E24A33', linestyle='--', label=f'Bias = {bias:.2f}')
    plt.axhline(loa_low, color='#59A14F', linestyle='--', label=f'Lower limit = {loa_low:.2f}')
    plt.axhline(loa_high, color='#59A14F', linestyle='--', label=f'Upper limit = {loa_high:.2f}')
    plt.axhline(0, color='#4C72B0', linewidth=1, alpha=0.5)
    plt.xlabel('Mean of Prediction and Reference (g/L)')
    plt.ylabel('Difference (Pred - True) (g/L)')
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    plt.close()


# Computes Hb range table
def hb_range_table(y_true, y_pred):
    # Define bins & labels
    bins = [-np.inf, 120, 150, np.inf]
    labels = ["<120 (anaemic)", "120–150 (normal)", ">150 (high)"]

    df = pd.DataFrame({
        "true": np.asarray(y_true).astype(float),
        "pred": np.asarray(y_pred).astype(float)
    })
    df["range"] = pd.cut(df["true"], bins=bins, labels=labels, right=False)

    rows = []
    for lab in labels:
        sub = df[df["range"] == lab]
        if len(sub) == 0:
            rows.append({
                "Hb Range": lab, "N": 0, "Mean True Hb (g/L)": np.nan,
                "MAE (g/L)": np.nan, "RMSE (g/L)": np.nan, "Bias (g/L)": np.nan
            })
            continue

        mae = mean_absolute_error(sub["true"], sub["pred"])
        rmse = np.sqrt(mean_squared_error(sub["true"], sub["pred"]))
        bias = float(np.mean(sub["pred"] - sub["true"]))
        rows.append({
            "Hb Range": lab,
            "N": int(len(sub)),
            "Mean True Hb (g/L)": float(np.mean(sub["true"])),
            "MAE (g/L)": float(mae),
            "RMSE (g/L)": float(rmse),
            "Bias (g/L)": float(bias)
        })

    out = pd.DataFrame(rows, columns=["Hb Range","N","Mean True Hb (g/L)","MAE (g/L)","RMSE (g/L)","Bias (g/L)"])
    return out

# Makes sure to use the right model
def tune_model(name, X, y, cv_folds, groups, registry):  
    return registry[name](X, y, cv_folds=cv_folds, groups=groups) 

# Makes predictions for each nail and makes a fused evaluation
def evaluate_fusion(
    base_model,
    features_per_nail,
    y,
    mask,
    cv_folds=n_folds,
    sanquin_features_per_nail=None, 
    sanquin_y=None 
):
    group_cv = GroupKFold(n_splits=cv_folds)     
    y_train = y[mask]
    val_preds, test_preds, train_preds = [], [], []

    # Only use Sanquin data if there is any (mostly for CNN)
    has_sanquin = (sanquin_y is not None)
    sanquin_preds = [] if has_sanquin else None

    # Takes each nail and uses a clone of tuned parameters
    for nail, features in enumerate(features_per_nail):                
        X_train_nail = features[mask]                                  
        groups_nail = X_train_nail.index.values 
        model = clone(base_model)

        # Predictions from validation 
        y_pred_val_nail = cross_val_predict(
            model, X_train_nail, y=y_train, groups=groups_nail, cv=group_cv
        )
        val_preds.append(y_pred_val_nail)                              

        # Early stopping for model fit XGBoost only 
        earlystop_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_es_idx, earlystop_idx = next(
            earlystop_split.split(X_train_nail, y_train, groups=groups_nail)
        )
        X_train_es, X_earlystop = X_train_nail.iloc[tr_es_idx], X_train_nail.iloc[earlystop_idx]
        y_train_es, y_earlystop = y_train.iloc[tr_es_idx], y_train.iloc[earlystop_idx]

        # Checks if the model is xgboost
        if isinstance(model.named_steps["regressor"], XGBRegressor):
            # Do early stopping if RMSE doesn't improve and fit on early stopping sets
            reg = model.named_steps["regressor"]
            reg.set_params(early_stopping_rounds=15, n_estimators=5000, verbosity=0)
            model.fit(
                X_train_es, y_train_es,
                regressor__eval_set=[(X_earlystop, y_earlystop)],
                regressor__verbose=False
            )
            # Get best number of trees
            best_iter = model.named_steps["regressor"].best_iteration
            final_model = clone(base_model)
            final_model.named_steps["regressor"].set_params(
                n_estimators=(best_iter + 1) if best_iter is not None
                            else reg.get_params()["n_estimators"],
                early_stopping_rounds=None,
                verbosity=0
            )
            # Refit for best model
            best_model = final_model.fit(X_train_nail, y_train)
        else:
            # Best model fit for other models 
            best_model = model.fit(X_train_nail, y_train)

        # Predictions of train data
        y_pred_train_nail = best_model.predict(X_train_nail)
        train_preds.append(y_pred_train_nail)

        # Predictions of test data
        X_test_nail = features[~mask]
        y_pred_test_nail = best_model.predict(X_test_nail)
        test_preds.append(y_pred_test_nail)

        # Predictions on Sanquin data only if they are available
        if has_sanquin:
            X_sanquin_nail = sanquin_features_per_nail[nail]
            y_pred_sanquin_nail = best_model.predict(X_sanquin_nail)
            sanquin_preds.append(y_pred_sanquin_nail)

    # Fuses the predictions of different nails together  
    y_pred_val   = np.mean(np.vstack(val_preds), axis=0)
    y_pred_train = np.mean(np.vstack(train_preds), axis=0)
    y_pred_test  = np.mean(np.vstack(test_preds), axis=0)

    # Metrics
    train_rmse, train_mae = compute_rmse_mae(y_train, y_pred_train)
    val_rmse, val_mae     = compute_rmse_mae(y_train, y_pred_val)
    y_test = y[~mask]
    test_rmse, test_mae   = compute_rmse_mae(y_test, y_pred_test)

    # Print the evaluation metrics
    print(f"Train:      RMSE={train_rmse:.2f} | MAE={train_mae:.2f}  (g/L)")
    print(f"Validation: RMSE={val_rmse:.2f} | MAE={val_mae:.2f}  (g/L)")
    print(f"Test:       RMSE={test_rmse:.2f} | MAE={test_mae:.2f}  (g/L)")

    # Sanquin metrics
    sanquin_rmse = sanquin_mae = None
    y_pred_sanquin = None

    if has_sanquin:
        y_pred_sanquin = np.mean(np.vstack(sanquin_preds), axis=0)
        sanquin_rmse, sanquin_mae = compute_rmse_mae(sanquin_y, y_pred_sanquin)
        print(f"Sanquin:   RMSE={sanquin_rmse:.2f} | MAE={sanquin_mae:.2f}  (g/L)")

    return {
        "train_rmse": train_rmse, "train_mae": train_mae,
        "val_rmse": val_rmse, "val_mae": val_mae,
        "test_rmse": test_rmse, "test_mae": test_mae,
        "y_train": y_train.values if hasattr(y_train, "values") else y_train,
        "y_pred_train": y_pred_train,
        "y_val": y_train.values if hasattr(y_train, "values") else y_train,
        "y_pred_val": y_pred_val,
        "y_test": y_test.values if hasattr(y_test, "values") else y_test,
        "y_pred_test": y_pred_test,
        "sanquin_rmse": sanquin_rmse,
        "sanquin_mae": sanquin_mae,
        "y_sanquin": (
            sanquin_y.values if (sanquin_y is not None)
            else sanquin_y),
        "y_pred_sanquin": y_pred_sanquin,
    }

# Algorithms -------------------------------------------------------------------------------------------------------------------
def run_elasticnet(X, y, cv_folds=n_folds, groups=None): 
    pipe = Pipeline([
        ("scaler",   RobustScaler()),
        ("regressor", ElasticNet(max_iter=10000))
    ])
    param_grid = {
        "regressor__l1_ratio": [0.01, 0.1, 0.5, 0.9, 0.99],
        "regressor__alpha": np.logspace(-4, 4, num=100),
    }
    search = GridSearchCV(
        pipe, param_grid, cv=GroupKFold(n_splits=cv_folds), 
        scoring="neg_root_mean_squared_error",
        return_train_score=True, verbose=1
    )
    search.fit(X, y, groups=groups) 
    print(f"[elasticnet] best CV RMSE: {-search.best_score_:.2f} g/L")
    print(f"[elasticnet] best params : {search.best_params_}")
    return search.best_estimator_

def run_xgboost(X, y, cv_folds=n_folds, groups=None): 
    pipe = Pipeline([
        ("regressor", XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1500,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1
        ))
    ])
    param_grid = {
        'regressor__learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'regressor__max_depth':         [3, 4],
        'regressor__subsample':         [0.6, 0.7, 0.8],
        'regressor__colsample_bytree':  [0.6,0.7, 0.8],
        'regressor__min_child_weight':  [1, 3, 5]
    }
    search = RandomizedSearchCV(
        pipe, param_grid,
        scoring='neg_root_mean_squared_error',
        n_iter=50,
        cv=GroupKFold(n_splits=cv_folds), 
        verbose=1,
        random_state=seed,
        n_jobs=-1
    )
    search.fit(X, y, groups=groups) 
    print(f"[XGBoost] best CV RMSE: {-search.best_score_:.2f} g/L")
    print(f"[XGBoost] best params : {search.best_params_}")
    return search.best_estimator_

def run_random_forest(X, y, cv_folds=n_folds, groups=None):
    pipe = Pipeline([
        ("regressor", RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            bootstrap=True,
            oob_score=False,
            criterion="absolute_error"
        ))
    ])
    param_grid = {
        "regressor__n_estimators":      [150, 200, 250],
        "regressor__min_samples_split": [9,10,11],
        "regressor__min_samples_leaf":  [3,4,5],
        "regressor__max_features":      ["sqrt", "log2", 0.3, 0.4],
        "regressor__max_depth":         [ 3, 4, 5]     
    }

    search = RandomizedSearchCV(
        pipe, param_grid,
        scoring='neg_root_mean_squared_error',
        n_iter=50,
        cv=GroupKFold(n_splits=cv_folds), 
        verbose=1,
        random_state=seed,
        n_jobs=-1
    )
    search.fit(X, y, groups=groups) 
    print(f"[Random Forest] best CV RMSE: {-search.best_score_:.2f} g/L")
    print(f"[Random Forest] best params : {search.best_params_}")
    return search.best_estimator_

def run_hybrid_cnn_rf(X, y, cv_folds=n_folds, groups=None):
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("selector", SelectKBest(f_regression)), 
        ("regressor", RandomForestRegressor(
            random_state=seed, 
            n_jobs=-1, 
            bootstrap=True
        ))
    ])
    param_grid = {
        "selector__k":                  [10, 15, 20], 
        "regressor__n_estimators":      [150, 200, 250],
        "regressor__min_samples_split": [9,10,11],
        "regressor__min_samples_leaf":  [3,4,5],
        "regressor__max_features":      ["sqrt", "log2", 0.3, 0.4],
        "regressor__max_depth":         [ 3, 4, 5]
    }   
    search = RandomizedSearchCV(
        pipe, param_grid,
        scoring='neg_root_mean_squared_error',
        n_iter=50,
        cv=GroupKFold(n_splits=cv_folds), 
        verbose=1,
        random_state=seed,
        n_jobs=-1
    )
    search.fit(X, y, groups=groups)
    print(f"[CNN+RF] best CV RMSE: {-search.best_score_:.2f} g/L")
    print(f"[CNN+RF] best params : {search.best_params_}")
    return search.best_estimator_

# Main ------------------------------------------------------------------------------------------------------------------------
def main():
    # Preprocess benchmark data
    features_per_nail, y, mask = preprocess_data(
        metadata, image_folder, n_nails=n_nails
    )
    nail_idx_for_tuning = 0 if len(features_per_nail) == 1 else 1

    sanquin_features_per_nail = None
    y_sanquin = None
    # Preprocess Sanquin data
    if sanquin_run:
        sanquin_features_per_nail, y_sanquin = preprocess_sanquin(
        sanquin_hb_csv,
        sanquin_image_folder,
        n_nails=n_nails
        )

    if sanquin_run:
        plot_features_benchmark_vs_sanquin(
            features_per_nail,     
            sanquin_features_per_nail,
            nail_idx_for_tuning, 
            sanquin_results_folder
        )


    # If EDA is turned on, it is started
    if eda_run:
            print("\n[EDA] Running exploratory data analysis on benchmark dataset")
            run_eda(features_per_nail, y, results_folder)

            # Percentile plot
            plot_rgb_percentiles_benchmark(features_per_nail, results_folder)

            if sanquin_run: 
                print("\n[EDA] Running exploratory data analysis on Sanquin dataset")
                run_eda_sanquin(sanquin_hb_csv, sanquin_image_folder, sanquin_results_folder)

    # If the hybrid is used, features are made
    cnn_features_per_nail = None
    if any(name == "CNN + RF hybrid" for name in models_run):
        if cnn_images_per_nail is None:
            raise RuntimeError("cnn_images_per_nail not prepared.")
        print("\n[cnn] Extracting DenseNet169 features")
        cnn_features_per_nail = extract_cnn_features_per_nail(cnn_images_per_nail, batch_size=32)
        for i in range(len(cnn_features_per_nail)):
            cnn_features_per_nail[i] = cnn_features_per_nail[i].loc[features_per_nail[i].index]

    # Extracts data for parameter tuning on second nail
    X_train_nail2 = features_per_nail[nail_idx_for_tuning][mask]
    y_train = y[mask]
    train_patient_ids = X_train_nail2.index.values 

    registry = {
        "Elastic Net": run_elasticnet,
        "XGBoost": run_xgboost,
        "Random Forest": run_random_forest,
        "CNN + RF hybrid": run_hybrid_cnn_rf,
    }

    summaries = []
    for name in models_run:
        print("\n" + "="*50)
        print(f"Running model: {name}")
        print("="*50)

        # Use CNN features if CNN is used
        if name == "CNN + RF hybrid":
            X_train_nail2 = cnn_features_per_nail[nail_idx_for_tuning][mask]
            train_patient_ids = X_train_nail2.index.values
            features_for_eval = cnn_features_per_nail

        else:
            features_for_eval = features_per_nail

        # Hyperparameter tuning only on the second nail 
        base_model = tune_model(name, X_train_nail2, y_train, n_folds, train_patient_ids, registry)

        # Evaluation on all nails with a fusion
        if sanquin_run and name != "CNN + RF hybrid":
            metrics = evaluate_fusion(
                base_model,
                features_for_eval,
                y,
                mask,
                cv_folds=n_folds,
                sanquin_features_per_nail=sanquin_features_per_nail,
                sanquin_y=y_sanquin,
            )
        else:
            metrics = evaluate_fusion(
                base_model,
                features_for_eval,
                y,
                mask,
                cv_folds=n_folds,
            )

        if error_run:
            # Creating Bland-Altman plots on test set
            y_true_test = metrics["y_test"]
            y_pred_test = metrics["y_pred_test"]
            plot_bland_altman(
                y_true_test, y_pred_test,
                title=f"Bland–Altman: {name}",
                savepath=os.path.join(results_folder, f"bland_altman_{name.replace(' ', '_')}_test.png")
            )
            # Creating Hb-range table on test set
            hb_table = hb_range_table(y_true_test, y_pred_test)
            print("\nHb-range error table for benchmark")
            print(hb_table.to_string(index=False))

            if sanquin_run and metrics["y_sanquin"] is not None:
                    y_true_sanquin = metrics["y_sanquin"]
                    y_pred_sanquin = metrics["y_pred_sanquin"]

                    # Bland–Altman for Sanquin, using the same function
                    plot_bland_altman(
                        y_true_sanquin, y_pred_sanquin,
                        title=f"Bland–Altman: {name} (Sanquin)",
                        savepath=os.path.join(
                            sanquin_results_folder,
                            f"bland_altman_{name.replace(' ', '_')}_sanquin.png"
                        )
                    )

                    # Hb-range table for Sanquin, using the same function
                    hb_table_sanq = hb_range_table(y_true_sanquin, y_pred_sanquin)
                    print("\nHb-range error table for Sanquin")
                    print(hb_table_sanq.to_string(index=False))
           
        summaries.append((name, metrics))

    print("\n--- Summary -----")
    for name, m in summaries:
        print(f"{name:>15} | "
              f"Train RMSE {m['train_rmse']:.2f}  MAE {m['train_mae']:.2f} | "
              f"Val RMSE {m['val_rmse']:.2f}  MAE {m['val_mae']:.2f} | "
              f"Test RMSE {m['test_rmse']:.2f}  MAE {m['test_mae']:.2f}")
        if m["sanquin_rmse"] is not None:
            print(f"{'':>15} | Sanquin RMSE {m['sanquin_rmse']:.2f}  MAE {m['sanquin_mae']:.2f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("ERROR:", e)
        traceback.print_exc()
        raise