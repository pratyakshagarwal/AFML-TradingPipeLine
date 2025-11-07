import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):
    """
    Extend KFold to work with labels that span time intervals.
    The training set is purged of observations that overlap with the test-label intervals.
    The test set is assumed contiguous (shuffle=False), without training samples in between.
    """

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates (t1) must be a pandas Series.")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_ranges = [(idx[0], idx[-1] + 1) for idx in np.array_split(indices, self.n_splits)]

        for i, j in test_ranges:
            t0 = self.t1.index[i]  # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())

            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)

            if maxT1Idx < X.shape[0]:  # right train (with embargo)
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))

            yield train_indices, test_indices


def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None,
    pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[train,:],y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values)
    if scoring=='neg_log_loss':
        prob=fit.predict_proba(X.iloc[test,:])
        score_=-log_loss(y.iloc[test],prob,
            sample_weight=sample_weight.iloc[test].values,labels=clf.classes_)
    else:
        pred=fit.predict(X.iloc[test,:])
        score_=accuracy_score(y.iloc[test],pred,sample_weight= \
                sample_weight.iloc[test].values)
    score.append(score_)
    return np.array(score)

if __name__ == '__main__':pass