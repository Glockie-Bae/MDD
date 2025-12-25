import numpy as np
from scipy.signal import welch
import scipy
import json
import torchaudio
import scipy.io as sio
import torch
import tqdm
import pandas as pd

def compute_psd(eeg, fs = 250):
    freqs, psd = welch(eeg, fs=fs, nperseg=fs*2)
    return freqs, psd

def band_power(psd, freqs, fmin, fmax):
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[:, mask].mean(axis=1)

def mfcc_stats(mfcc):
    mean = mfcc.mean(axis=1)
    std  = mfcc.std(axis=1)
    skew = scipy.stats.skew(mfcc, axis=1)
    return np.concatenate([mean, std, skew], axis=1)


def eeg_data(path):
    eeg128_data = sio.loadmat(path)
    eeg_key = None
    for k, v in eeg128_data.items():
        if (
                not k.startswith('__')
                and isinstance(v, np.ndarray)
                and v.ndim == 2
                and v.shape[0] == 129
        ):
            eeg_key = k
            break
    eeg128_data = eeg128_data[eeg_key]

    eeg128 = eeg128_data[:128, :]

    alpha_at_11_5 = [11, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0]

    freqs, psd = compute_psd(eeg128)

    theta = band_power(psd, freqs, 4, 8)
    alpha = band_power(psd, freqs, 8, 13)
    beta = band_power(psd, freqs, 13, 20)

    eeg_features = []
    eeg_features.append(theta.mean(axis=0))
    eeg_features.append(alpha.mean(axis=0))
    eeg_features.append(beta.mean(axis=0))
    eeg_features.append(11.5)
    return eeg_features


def wav_data(filename, data_mode):
    # mixup
    filename = r"F:\_Sorrow\PhD\王园园\AMAST-main\src"  + filename[1:]
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()


    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0,
                                              frame_shift=10)

    target_length = 512
    n_frames = fbank.shape[0]

    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    mix_lambda = 0

    if data_mode == "train":
        freqm = torchaudio.transforms.FrequencyMasking(0)
        timem = torchaudio.transforms.TimeMasking(0)
    elif data_mode == "test":
        freqm = torchaudio.transforms.FrequencyMasking(0)
        timem = torchaudio.transforms.TimeMasking(0)
    fbank = torch.transpose(fbank, 0, 1)
    # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
    fbank = fbank.unsqueeze(0)
    if freqm != 0:
        fbank = freqm(fbank)
    if timem != 0:
        fbank = timem(fbank)
    # squeeze it back, it is just a trick to satisfy new torchaudio version
    fbank = fbank.squeeze(0)
    fbank = torch.transpose(fbank, 0, 1)

    # normalize the input for both training and test
    fbank = (fbank - -4.2677393) / (4.5689974 * 2)

    return fbank


def read_data(data_json, data_mode="train"):
    EEG_X = []
    Audio_x = []
    labels = []

    index = 0
    for data in data_json:
        eeg_path = data['eeg128'][0]
        label = data['labels']
        wav_path = data['wav']
        eeg_features = eeg_data(eeg_path)
        wav_features = wav_data(wav_path, data_mode)
        wav_features = mfcc_stats(wav_features)

        EEG_X.append(eeg_features)
        Audio_x.append(wav_features)
        labels.append(label)
        print(f"finished {index} data")
        index += 1

    np.save(f"EEG_X_{data_mode}.npy", EEG_X)
    np.save(f"Audio_X_{data_mode}.npy", Audio_x)
    np.save(f"label_{data_mode}.npy", labels)
    return EEG_X, Audio_x, labels

def mfcc_stats(mfcc):
    mean = mfcc.mean(axis=0)
    std  = mfcc.std(axis=0)
    skew = scipy.stats.skew(mfcc, axis=0)
    return np.concatenate([np.array(mean), np.array(std), skew], axis=0)

# train_json_path = r"F:\_Sorrow\PhD\王园园\AMAST-main\src\pre-processing\data-2-0.8-rst\datafiles\lanzhou_audio_train.json"
# test_json_path = r"F:\_Sorrow\PhD\王园园\AMAST-main\src\pre-processing\data-2-0.8-rst\datafiles\lanzhou_audio_test.json"
#
# with open(train_json_path, 'r') as fp:
#     train_json = json.load(fp)
# train_data = train_json['data']
#
# with open(test_json_path, 'r') as fp:
#     test_json = json.load(fp)
# test_data = test_json['data']

# EEG_X_test,  Audio_X_test, labels_test = read_data(test_data, "test")
# EEG_X_train, Audio_X_train, labels_train = read_data(train_data, "train")

data_root = r"F:\_Sorrow\PhD\材料\yiyun_py\MultiModal"
EEG_X_train = np.load(data_root + r"\EEG_X_train.npy")
EEG_X_test = np.load(data_root + r"\EEG_X_test.npy")
Audio_X_train = np.load(data_root + r"\Audio_X_train.npy")
Audio_X_test = np.load(data_root + r"\Audio_X_test.npy")
label_train = np.load(data_root + r"\label_train.npy")
label_test = np.load(data_root + r"\label_test.npy")

X_fusion_train = np.concatenate([EEG_X_train, Audio_X_train], axis=1)
X_fusion_test = np.concatenate([EEG_X_test, Audio_X_test], axis=1)
print(f"X_fusion_train shape is :{X_fusion_train.shape}")
print(f"X_fusion_test shape is : {X_fusion_test.shape}")

for index, label in enumerate(label_train):
    if label[-2:] == '00':
        label_train[index] = 0
    elif label[-2:] == '01':
        label_train[index] = 1
for index, label in enumerate(label_test):
    if label[-2:] == '00':
        label_test[index] = 0
    elif label[-2:] == '01':
        label_test[index] = 1

label_train = np.array(label_train, dtype=int)
label_test = np.array(label_test, dtype=int)
#ROC
#import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
# Split data into train/test
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# XGBoost模型参数
params_xgb = {
    'learning_rate': 0.02,
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'max_leaves': 127,
    'verbosity': 1,
    'seed': 42,
    'nthread': -1,
    'colsample_bytree': 0.6,
    'subsample': 0.7,
    'n_estimators': 200,  # 固定树的数量
    'max_depth': 4,       # 固定树的深度
    'min_child_weight': 1 # 固定节点最小权重
}

# Define models
models_dict = {
    "RandomForest": Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42,max_depth=10 , max_features = 'log2', n_estimators = 100,
                                             min_samples_split = 2, min_samples_leaf = 2))
    ]),
    "GradientBoosting": Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42,learning_rate = 0.2, max_depth = 7, n_estimators = 200, subsample = 0.8))
    ]),
    "LightGBM": Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LGBMClassifier(random_state=42, verbose=-1))
    ]),
    "CatBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', CatBoostClassifier(verbose=0, random_state=42,depth = 6,iterations = 100, learning_rate =0.05))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='linear', probability=True, max_iter=10000, class_weight='balanced', C=0.001, gamma=0.0001, random_state=42))
    ])
}

# print("是否存在 NaN:", np.isnan(X_fusion_train).any())
# print("NaN 总数:", np.isnan(X_fusion_train).sum())
#
# print("是否存在 NaN:", np.isnan(X_fusion_test).any())
# print("NaN 总数:", np.isnan(X_fusion_test).sum())
#
# nan_rows = np.isnan(X_fusion_train).any(axis=0)
# print(np.where(nan_rows)[0])
#
# nan_rows = np.isnan(X_fusion_test).any(axis=0)
# print(np.where(nan_rows)[0])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model_results = {}
# Cross-validation loop
for model_name, model in tqdm_notebook(models_dict.items()):
    fold_roc_data = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_fusion_train, label_train)):
        # Split data into train and test
        X_train_fold, X_test_fold = X_fusion_train[train_idx], X_fusion_train[test_idx]
        y_train_fold, y_test_fold = label_train[train_idx], label_train[test_idx]


        # Train the model
        model.fit(X_train_fold, y_train_fold)

        # Get predicted probabilities
        y_proba = model.predict_proba(X_test_fold)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test_fold, y_proba)
        roc_auc = auc(fpr, tpr)

        # Save the fold ROC data
        fold_roc_data.append(pd.DataFrame({
            'fold': fold_idx + 1,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'model': model_name
        }))

    # Concatenate fold ROC data for the current model
    model_results[model_name] = pd.concat(fold_roc_data, ignore_index=True)