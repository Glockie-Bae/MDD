import pandas as pd
from openai.cli import Audio
import numpy as np
import glob
import scipy.io as sio
from scipy.signal import welch
from scipy.signal import butter, filtfilt

def extract_common_rows_exact(df1, df2):
    """
    提取两个 DataFrame 中完全相同的行
    """
    return pd.merge(df1, df2, how="inner")



def bandpass_filter(eeg, fs, low=1, high=50, order=4):
    """
    eeg: [n_channels, n_samples]
    fs : sampling rate
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, eeg, axis=-1)

from scipy.signal import iirnotch

def notch_filter(eeg, fs, freq=50, Q=30):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, eeg, axis=-1)


def compute_relative_power(eeg, fs=250, fmin=4, fmax=20, df=0.1):
    """
    eeg: [channels, time]
    return: [freq_bins]
    """

    eeg = bandpass_filter(eeg, fs=250, low=1, high=50)
    eeg = notch_filter(eeg, fs=250, freq=50)


    freqs, psd = welch(eeg, fs=fs, nperseg=fs * 2, axis=-1)

    # 选频段
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs_sel = freqs[mask]
    psd_sel = psd[:, mask]

    # 插值到 0.1 Hz
    target_freqs = np.arange(fmin, fmax + df, df)
    psd_interp = np.array([
        np.interp(target_freqs, freqs_sel, psd_sel[ch])
        for ch in range(psd_sel.shape[0])
    ])

    # 额区通道平均
    psd_mean = psd_interp.mean(axis=0)

    # 相对功率
    rel_power = psd_mean / psd_mean.sum()

    return target_freqs, rel_power, freqs

def read_multimodal_data(df, EEG128_ROOT):
    data = []
    for _, row in df.iterrows():
        subj_id = row['subject id']
        gender = row['gender']
        label = 1 if row['type'] == 'MDD' else 0

        # eeg128_dir = EEG128_ROOT / subj_id
        # eeg3_dir = EEG3_ROOT / subj_id

        eeg128_data_path = glob.glob(EEG128_ROOT + f"\*{subj_id}*")[0]
        #eeg3_data_path = glob.glob(EEG3_ROOT + f"\*{subj_id}*")[0]

        # if len(eeg128_data_path) == 0 or len(eeg3_data_path) == 0:
        #     print(f"[WARN] 缺失被试 {subj_id}")
        #     continue

        # 举例：读取 EC 状态
        #eeg3_data = np.loadtxt(eeg3_data_path)
        eeg128_data = sio.loadmat(eeg128_data_path)
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
        #eeg3 = np.transpose(eeg3_data, (1, 0))

        #target_freqs, rp3, freqs = compute_relative_power(eeg3)
        _, rp128, _ = compute_relative_power(eeg128)
        #rp128_dir = r"F:\_Sorrow\PhD\材料\EEG_128channels_resting_lanzhou_2015\rp"
        #np.save(f"{rp128_dir}\\{subj_id}_rp.npy", rp128)

        data.append({
            "subject_id": subj_id,
            'gender': gender,
            "label": label,
            # "eeg128": eeg128_data,
            # "eeg3": eeg3_data,
            # 'freqs': freqs,
            #'rp_3ch': rp3,
            'rp_128ch': rp128,
            'PHQ9': row['PHQ-9'],
            'GAD7': row['GAD-7'],
            'CTQ': row['CTQ-SF'],
            'age': row['age'],
        })
        print(data)
    return data

Audio_data_dir = r"F:\_Sorrow\PhD\材料\audio_lanzhou_2015"
EEG_128_rest_data_dir = f"F:\_Sorrow\PhD\材料\EEG_128channels_resting_lanzhou_2015"
EEG_3_data_dir = f"F:\_Sorrow\PhD\材料\EEG_3channels_resting_lanzhou_2015"


EEG_128_rest_subj_inf_path = EEG_128_rest_data_dir + "\\subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"
EEG_128_rest_subj_inf = pd.read_excel(EEG_128_rest_subj_inf_path, usecols="A,B,C,D,E,F,G,H,I,J,K") # 默认读取第一个sheet

Audio_subj_inf_path = Audio_data_dir + "\\subjects_information_audio_lanzhou_2015.xlsx"
Audio_subj_inf = pd.read_excel(Audio_subj_inf_path, usecols="A,B,C,D,E,F,G,H,I,J,K") # 默认读取第一个sheet

EEG_3_subj_inf_path = EEG_3_data_dir + "\\subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx"
EEG_3_subj_inf = pd.read_excel(EEG_3_subj_inf_path, usecols="A,B,C,D,E,F,G,H,I,J,K") # 默认读取第一个sheet

subj_inf = extract_common_rows_exact(EEG_128_rest_subj_inf, Audio_subj_inf)
print(subj_inf)

data = read_multimodal_data(subj_inf, EEG_128_rest_data_dir)
np.savez('egg128_audio_dataset.npz', data)