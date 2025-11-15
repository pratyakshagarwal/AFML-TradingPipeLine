import pandas as pd
import numpy as np
from scipy.stats import weightedtau

def get_eVec(dot,varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal,eVec=np.linalg.eigh(dot)
    idx=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[idx],eVec[:,idx]
    #2) only positive eVals
    eVal=pd.Series(eVal,index=['PC_'+str(i+1) for i in range(eVal.shape[0])])
    eVec=pd.DataFrame(eVec,index=dot.index,columns=eVal.index)
    eVec=eVec.loc[:,eVal.index]
    #3) reduce dimension, form PCs
    cumVar=eVal.cumsum()/eVal.sum()
    dim=cumVar.values.searchsorted(varThres)
    eVal,eVec=eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
    return eVal,eVec


#-----------------------------------------------------------------
def orthoFeats(dfX,varThres=.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dfZ=dfX.sub(dfX.mean(),axis=1).div(dfX.std(),axis=1) # standardize
    dot=pd.DataFrame(np.dot(dfZ.T,dfZ),index=dfX.columns,columns=dfX.columns)
    eVal,eVec=get_eVec(dot,varThres)
    dfP=np.dot(dfZ,eVec)
    return dfP, eVal, eVec

#---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weightedtau

def compare_imp(eVal, mdi, save_path=None):
    # Compare PCA eigenvalue variance (unsupervised) with MDI importances (supervised)
    # to assess if the model is learning real structure or noise.
    # --- Sanity checks ---
    eVal = eVal / eVal.sum()  # normalize to represent variance ratio
    mdi = mdi / mdi.sum()     # normalize to comparable scale

    # --- Align dimensions ---
    n = min(len(eVal), len(mdi))
    eVal = eVal.iloc[:n]
    mdi = mdi.iloc[:n]

    # --- Weighted Kendall’s Tau correlation ---
    tau, p_value = weightedtau(eVal, mdi)
    print(f"Weighted Kendall’s Tau between PCA variance and MDI importance: {tau:.4f} (p={p_value:.4f})")

    # --- Plot comparison ---
    plt.figure(figsize=(8, 5))
    plt.scatter(eVal, mdi, alpha=0.8, color='darkorange')
    plt.plot([0, max(eVal.max(), mdi.max())], [0, max(eVal.max(), mdi.max())],
             linestyle='--', color='gray', alpha=0.5)
    plt.xlabel("Explained Variance (PCA eigenvalues)")
    plt.ylabel("MDI Importance")
    plt.title(f"PCA Variance vs MDI Importance (Weighted τ={tau:.3f})")
    plt.grid(alpha=0.3)

    if save_path:
        plt.tight_layout()
        path = f"{save_path}/pca_vs_mdi.png"
        plt.savefig(path, dpi=300)
    else:
        plt.show()

    return path
