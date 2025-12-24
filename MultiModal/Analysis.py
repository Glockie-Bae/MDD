import numpy as np
from scipy.signal import welch
from pingouin import intraclass_corr
from scipy.stats import pearsonr


results = np.load("egg_dataset.npz", allow_pickle=True)['arr_0']
freqs = np.arange(4, 20.1, 0.1)


rp3_all = np.stack([r['rp_3ch'] for r in results])
rp128_all = np.stack([r['rp_128ch'] for r in results])

# 整体 Pearson
r, p = pearsonr(rp3_all.mean(1), rp128_all.mean(1))
print(f'Pearson r={r:.3f}, p={p:.4e}')


from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

male = rp3_all[[r['gender']=='M' for r in results]]
female = rp3_all[[r['gender']=='F' for r in results]]

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
    y = np.array([r[scale] for r in results])
    r_val, p_val = pearsonr(alpha_power, y)
    print(scale, r_val, p_val)

from sklearn.metrics import roc_auc_score

labels = np.array([r['label'] for r in results])
auc = roc_auc_score(labels, alpha_power)
print("ROC-AUC:", auc)


# ================绘制个体差异=======================
import matplotlib.pyplot as plt
import numpy as np

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

# ================绘制性别差异=======================
# genders = results['gender'].values   # length = N_subjects
#
#
# male_idx   = genders == 'M'
# female_idx = genders == 'F'
#
# mean_male = rp3_all[male_idx].mean(axis=0)
# std_male  = rp3_all[male_idx].std(axis=0)
#
# mean_fem = rp3_all[female_idx].mean(axis=0)
# std_fem  = rp3_all[female_idx].std(axis=0)
#
# plt.figure(figsize=(7,5))
#
# plt.plot(freqs, mean_male, color='blue', label='Male')
# plt.fill_between(freqs,
#                  mean_male - std_male,
#                  mean_male + std_male,
#                  color='blue', alpha=0.3)
#
# plt.plot(freqs, mean_fem, color='red', label='Female')
# plt.fill_between(freqs,
#                  mean_fem - std_fem,
#                  mean_fem + std_fem,
#                  color='red', alpha=0.3)
#
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Relative Power')
# plt.legend()
# plt.title('Gender-based spectral power (3-channel)')
# plt.show()
