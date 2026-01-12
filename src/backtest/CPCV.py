import itertools as it
import numpy as np

class CombinatorialPurgedCV:
    def __init__(self, n_splits=6, n_test_splits=2, pct_embargo=0.01, purge_window=2):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0 <= pct_embargo < 1):
            raise ValueError("pct_embargo must be in [0,1)")
        if purge_window < 0:
            raise ValueError("purge_window must be >= 0")

        self.n_splits = int(n_splits)
        self.n_test_splits = int(n_test_splits)
        self.pct_embargo = float(pct_embargo)
        self.purge_window = int(purge_window)

    def _make_groups(self, T: int):
        """Split indices [0..T) into n_splits groups."""
        size = T // self.n_splits
        groups, start = [], 0
        for i in range(self.n_splits):
            end = start + size if i < self.n_splits - 1 else T
            groups.append(np.arange(start, end))
            start = end
        return groups

    def _mask(self, test_blocks, T: int):
        """Build boolean mask for embargo + purge logic."""
        mask = np.zeros(T, dtype=bool)
        for block in test_blocks:
            mask[block] = True
            imin, imax = block.min(), block.max() + 1

            if imin >= 0:
                left = max(0, imin - self.purge_window)
                mask[np.arange(left, imin)] = True

            if imax <= T:
                right = min(T, imax + self.purge_window + int(self.pct_embargo * T))
                mask[np.arange(imax, right)] = True

        return mask

    def split(self, X, y=None):
        """Yield (train_idx, test_idx) pairs."""
        T = len(X)
        if T <= 0:
            raise ValueError("X must be non-empty")

        groups = self._make_groups(T)
        grp_idx = np.arange(len(groups))
        combs = list(it.combinations(grp_idx, self.n_test_splits))

        for test_blocks in combs:
            test_idx = np.concatenate([groups[g] for g in test_blocks]) if test_blocks else np.array([], dtype=int)
            train_blocks = tuple(set(grp_idx) - set(test_blocks))
            if not train_blocks:
                continue

            # train_idx = np.concatenate([groups[g] for g in sorted(train_blocks)])
            mask = self._mask([groups[g] for g in test_blocks], T)

            final_train_idx = np.where(~mask)[0]
            final_test_idx = np.array(test_idx, dtype=int)

            assert len(final_train_idx) + len(final_test_idx) <= T, "Indices exceed dataset size"
            yield final_train_idx, final_test_idx


if __name__ == "__main__":
    X = np.linspace(1, 1000, 600).reshape(120, 5)
    y = np.random.randint(0, 5, 120)

    cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
    for tr, te in cv.split(X):
        print("TRAIN:", tr, "TEST:", te)
        print("TRAIN SIZE:", len(tr), "TEST SIZE:", len(te))
        print("-----")