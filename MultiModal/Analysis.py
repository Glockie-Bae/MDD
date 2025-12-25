import numpy as np
from scipy.signal import welch
from pingouin import intraclass_corr
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

def corr_per_freq(X, y):
    r_vals, p_vals = [], []
    for i in range(X.shape[1]):
        r, p = pearsonr(X[:, i], y)
        r_vals.append(r)
        p_vals.append(p)
    return np.array(r_vals), np.array(p_vals)

def gender_corr(col_name, rp = 3):
    col_all = np.stack([r[col_name] for r in results])
    male_col = col_all[[r['gender'] == 'M' for r in results]]
    female_col = col_all[[r['gender'] == 'F' for r in results]]

    if rp == 3:
        male = rp3_all[[r['gender'] == 'M' for r in results]]
        female = rp3_all[[r['gender'] == 'F' for r in results]]
    elif rp == 128:
        male = rp128_all[[r['gender'] == 'M' for r in results]]
        female = rp128_all[[r['gender'] == 'F' for r in results]]
    else:
        print("rp 输入错误，需要输入 3 或者 128")
        return


    r_phq_m, p_phq_m = corr_per_freq(male, male_col)
    rej_phq_m, p_phq_m_fdr = fdrcorrection(p_phq_m)


    r_phq_f, p_phq_f = corr_per_freq(female, female_col)
    rej_phq_f, p_phq_f_fdr = fdrcorrection(p_phq_f)
    # ===============画图===================
    plt.figure(figsize=(8, 4))

    plt.plot(freqs, r_phq_m, label='Male', color='red')
    plt.plot(freqs, r_phq_f, label='Female', color='blue')

    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Pearson r')
    plt.title(f'Correlation between {col_name} and Relative Spectral Power')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # ===============画图===================

    # ===== 整理显著频点表 =====
    sig_tables = {}

    for label, rej, rvals, pvals in [
        ('Male', rej_phq_m, r_phq_m, p_phq_m_fdr),
        ('Female', rej_phq_f, r_phq_f, p_phq_f_fdr)
    ]:
        sig_idx = np.where(rej)[0]

        if len(sig_idx) == 0:
            print(f"\n{col_name} | {label}: ❌ FDR 校正后无显著频点")
            sig_tables[label] = None
        else:
            df = pd.DataFrame({
                'Frequency(Hz)': freqs[sig_idx],
                'r': rvals[sig_idx],
                'p_fdr': pvals[sig_idx]
            })
            print(f"\n{col_name} | {label}: ✅ FDR 显著频点：")
            print(df)
            sig_tables[label] = df

    # 找最接近 11.9 Hz 的索引
    idx_119 = np.argmin(np.abs(freqs - 11.9))

    r_m, p_m = pearsonr(male[:, idx_119], male_col)
    r_f, p_f = pearsonr(female[:, idx_119], female_col)

    print(f"{col_name} feature 11.9 Hz Male:", r_m, p_m)
    print(f"{col_name} feature 11.9 Hz Female:", r_f, p_f)

results = np.load("egg_dataset.npz", allow_pickle=True)['arr_0']
freqs = np.arange(4, 20.1, 0.1)
label_all = np.stack([r['label'] for r in results])
rp3_all = np.stack([r['rp_3ch'] for r in results])
rp128_all = np.stack([r['rp_128ch'] for r in results])
PHQ_all = np.stack([r['PHQ9'] for r in results])

# 整体 Pearson
r, p = pearsonr(rp3_all.mean(0), rp128_all.mean(0))
print(f'Pearson r={r:.3f}, p={p:.4e}')



male = rp3_all[[r['gender']=='M' for r in results]]
female = rp3_all[[r['gender']=='F' for r in results]]
male_PHQ = PHQ_all[[r['gender']=='M' for r in results]]
female_PHQ = PHQ_all[[r['gender']=='F' for r in results]]

#===========探究性别与PHQ-9量表的相关性=======================


pvals = []
for i in range(rp3_all.shape[1]):
    _, p = mannwhitneyu(male[:, i], female[:, i])
    pvals.append(p)

pvals = np.array(pvals)
reject, p_fdr = fdrcorrection(pvals, alpha=0.05)
#print(f'Pearson r={reject:.3f}, p={p_fdr:.4e}')

alpha_mask = (freqs >= 11) & (freqs <= 14)
alpha_power = rp3_all[:, alpha_mask].mean(axis=1)

from scipy.stats import pearsonr

for scale in ['PHQ9', 'GAD7', 'CTQ']:
    gender_corr(scale, 128)

from sklearn.metrics import roc_auc_score

labels = np.array([r['label'] for r in results])
auc = roc_auc_score(labels, alpha_power)
print("ROC-AUC:", auc)


# ================绘制个体差异=======================

N = rp3_all.shape[0]

n_cols = 7
n_rows = int(np.ceil(N / n_cols))  # 自动算行数（40 → 6）

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(n_cols * 3, n_rows * 2.5),
    sharex=True,
    sharey=True
)

axes = axes.flatten()  # 拉平成一维，方便索引

for i in range(N):
    ax = axes[i]

    ax.plot(freqs, rp3_all[i], color='red', alpha=0.9, label='3-ch')
    ax.plot(freqs, rp128_all[i], color='blue', alpha=0.9, label='128-ch')

    ax.grid(alpha=0.3)

# 把多余的空子图关掉
for j in range(N, len(axes)):
    axes[j].axis('off')

# 统一坐标轴标签（只在外圈显示）
fig.text(0.5, 0.04, 'Frequency (Hz)', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Relative Power', va='center', rotation='vertical', fontsize=12)

# 统一图例（只放一个）
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.suptitle('Subject-wise Relative Power: 3-channel vs 128-channel EEG', fontsize=14)
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
plt.show()


# ================绘制群体差异=======================
mean_3  = rp3_all.mean(axis=0)
std_3   = rp3_all.std(axis=0)

mean_128 = rp128_all.mean(axis=0)
std_128  = rp128_all.std(axis=0)

plt.figure(figsize=(7,5))

plt.plot(freqs, mean_3, color='red', label='3-channel')
plt.fill_between(freqs,
                 mean_3 - std_3,
                 mean_3 + std_3,
                 color='red', alpha=0.3)

plt.plot(freqs, mean_128, color='blue', label='128-channel')
plt.fill_between(freqs,
                 mean_128 - std_128,
                 mean_128 + std_128,
                 color='blue', alpha=0.3)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Power')
plt.legend()
plt.title('Mean spectral power ± SD')
plt.show()

#================绘制性别差异=======================
male_rp3 = rp3_all[[r['gender']=='M' for r in results]]
female_rp3 = rp3_all[[r['gender']=='F' for r in results]]

mean_male = male_rp3.mean(axis=0)
std_male  = male_rp3.std(axis=0)

mean_fem = female_rp3.mean(axis=0)
std_fem  = female_rp3.std(axis=0)

plt.figure(figsize=(7,5))

plt.plot(freqs, mean_male, color='blue', label='Male')
plt.fill_between(freqs,
                 mean_male - std_male,
                 mean_male + std_male,
                 color='blue', alpha=0.3)

plt.plot(freqs, mean_fem, color='red', label='Female')
plt.fill_between(freqs,
                 mean_fem - std_fem,
                 mean_fem + std_fem,
                 color='red', alpha=0.3)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Power')
plt.legend()
plt.title('Gender-based spectral power (3-channel)')
plt.show()
