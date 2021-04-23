import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

sns.set()
sns.set_context('talk')

df = pd.read_csv('./ood_roc_auc_values.csv',
                 names=['ood_set', 'comparison_set', 'backbone', 'ood_class', 'score_fn', 'roc_auc'])

for backbone in ['resnet', 'mlp']:
    plt.figure(figsize=(10, 5))
    plt.plot([-2, 7], [0.5, 0.5], 'k-.', lw=1)

    ttest_res = ttest_ind(
        df[(df['backbone'] == backbone) & (df['score_fn'] == 'px_grad')]['roc_auc'].values,
        df[(df['backbone'] == backbone) & (df['score_fn'] == 'svm_cal')]['roc_auc'].values,
    )
    mw_res = mannwhitneyu(
        df[(df['backbone'] == backbone) & (df['score_fn'] == 'px_grad')]['roc_auc'].values,
        df[(df['backbone'] == backbone) & (df['score_fn'] == 'svm_cal')]['roc_auc'].values,
    )
    print(f'Man Whitney U test {mw_res} of pxgrad vs svm_cal for backbone {backbone}')
    print(f'T test {ttest_res} of pxgrad vs svm_cal for backbone {backbone}')

    sns.violinplot(x='score_fn', y='roc_auc', data=df[df['backbone'] == backbone], cut=0)
    sns.swarmplot(x='score_fn', y='roc_auc', data=df[df['backbone'] == backbone],
                  color='white', size=5, linewidth=1, edgecolor='black')
    plt.ylim(0, 1)
    plt.xlabel('\nscore function for OOD calculation')
    plt.ylabel('ROC area under the curve')
    plt.title(f'Comparison between OOD scoring functions for test vs OOD {backbone}')
    plt.tight_layout()
    plt.savefig(f'./figs/violin_plot_scoring_fnc_OOD_{backbone}.pdf')
