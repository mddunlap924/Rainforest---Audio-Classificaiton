import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

file0 = r'E:\Kaggle\RainForest_R0\HPTuning\Combinations\Feb 15_1\224_224_33\DenseNet121_SED\DenseNet121_0\clip'
file1 = r'E:\Kaggle\RainForest_R0\HPTuning\Combinations\Feb 15_1\224_224_53\DenseNet121_SED\DenseNet121_0\clip'
file2 = r'E:\Kaggle\RainForest_R0\HPTuning\Combinations\Feb 15_1\224_224_123\DenseNet121_SED\DenseNet121_0\clip'
save_path = r'E:\Kaggle\RainForest_R0\HPTuning\Combinations\Feb 15_1\DenseNet121\Rev5'
weights = [0.3, 0.35, 0.35]
sum(weights)

sub0 = pd.read_csv(os.path.join(file0, 'submission.csv'))
sub1 = pd.read_csv(os.path.join(file1, 'submission.csv'))
sub2 = pd.read_csv(os.path.join(file2, 'submission.csv'))

final_sub = sub1.copy()

sub0_ = sub0.iloc[:, 1:].to_numpy()
sub1_ = sub1.iloc[:, 1:].to_numpy()
sub2_ = sub2.iloc[:, 1:].to_numpy()

# final_sub_ = (weights[0] * sub0_) + (weights[1] * sub1_) + (weights[2] * sub2_)
final_sub_ = np.mean(np.array([sub0_, sub1_, sub2_]), axis=0)
gain = 1.3
final_sub_[:, 3] = np.clip(final_sub_[:, 3] * gain, 0.0, 1.0)
final_sub_[:, 18] = np.clip(final_sub_[:, 18] * gain, 0.0, 1.0)


high_score_path = r'C:\Kaggle\RainForest_R0\Forked Submissions\My Kaggle Submissions\Single Model\DenseNet121 Merge 0\clip'
# high_score_path = r'E:\Kaggle\RainForest_R0\HPTuning\Combinations\Feb 9\DenseNet121\Rev1\clip'
high_score = pd.read_csv(os.path.join(high_score_path, 'submission.csv'))
high_score = high_score.iloc[:, 1:].to_numpy()

# Loop through possible weights
weight_ranges = np.arange(0.00, 1.05, 0.05)
weight_scores = []
count = 0
for i, w0 in enumerate(weight_ranges):
    for j, w1 in enumerate(weight_ranges):
        for k, w2 in enumerate(weight_ranges):
            sum_check = np.sum(w0 + w1 + w2)
            if sum_check == 1.0:
                y_avg = (w0 * sub0_) + (w1 * sub1_) + (w2 * sub2_)
                abs_diff = np.sum(np.abs(high_score - y_avg))
                norm_diff = np.sum(high_score - y_avg)
                weight_scores.append([i, j, k, w0, w1, w2, abs_diff, norm_diff])
                count += 1
                print(f'Count {count}')

weight_scores = np.array(weight_scores)
plt.plot(weight_scores[:, 6])
plt.plot(weight_scores[:, 7])
plt.hlines(0, 0, len(weight_scores), colors='k')
plt.hlines(np.min(weight_scores[:, 6]), 0, len(weight_scores), colors='k')
score_diff = high_score - final_sub_

# final_sub_[:, -1] = high_score[:, -1]
final_sub.iloc[:, 1:] = final_sub_
fig, [ax0, ax1, ax2] = plt.subplots(1, 3, figsize=(15, 8))
im0 = ax0.imshow(high_score, aspect='auto')
ax0.set_title('High Score')
fig.colorbar(im0, ax=ax0)

im1 = ax1.imshow(final_sub_, aspect='auto')
ax1.set_title('Final Sub.')
fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow(score_diff, vmin=-0.2, vmax=0.2, aspect='auto')
ax2.set_title(f'Difference {score_diff.sum()}; {np.abs(score_diff).sum()}')
fig.colorbar(im2, ax=ax2)

plt.show()


final_sub.to_csv(os.path.join(save_path, 'submission.csv'), index=False)

print('check_point')
