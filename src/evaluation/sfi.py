import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.kFold import cvScore


def auxFeatImpSFI(featNames, clf, trnsX, y, sample_weight,
                pctEmbargo, scoring,cvGen=None, t1=None, cv=None):
    imp=pd.DataFrame(columns=['mean','std'])
    for featName in featNames:
        df0=cvScore(clf,X=trnsX[[featName]],y=y,sample_weight=sample_weight,
            scoring=scoring,cvGen=cvGen, pctEmbargo=pctEmbargo, t1=t1, cv=cv)
        imp.loc[featName,'mean']=df0.mean()
        imp.loc[featName,'std']=df0.std()*df0.shape[0]**-.5
    return imp

def map_featImpSfi(imp, folder_path):
    # Plot Single Feature Importance (SFI) mean ± std as a bar chart and save it.
    # Sort by mean importance descending for cleaner visualization
    imp_sorted = imp.sort_values("mean", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(imp_sorted.index, imp_sorted["mean"], yerr=imp_sorted["std"],
        capsize=4, color="steelblue", alpha=0.8, label="Mean ± Std"
    )

    plt.xticks(rotation=60, ha='right')
    plt.ylabel("SFI Importance")
    plt.title("Single Feature Importance (SFI)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = f"{folder_path}/sfi.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
