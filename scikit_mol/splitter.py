import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import pandas as pd
from collections import defaultdict
from typing import Union, List
from sklearn.model_selection._split import BaseShuffleSplit, _validate_shuffle_split
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_random_state
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import indexable
from sklearn.utils._array_api import ensure_common_namespace_device
from itertools import chain
from sklearn.utils import _safe_indexing


class StratifiedGroupShuffleSplit(BaseShuffleSplit):
    """Stratified ShuffleSplit cross-validator with non-overlapping groups."""

    def __init__(
        self,
        n_splits: int = 5,
        *,
        test_size: float or int = 0.2,
        train_size: float or int=None,
        random_state: int = None,
        sample_weighted: bool = False,
        suppress_warnings: bool = False
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.sample_weighted = sample_weighted
        self.suppress_warnings = suppress_warnings

        if not suppress_warnings:
            if self.sample_weighted:
                warnings.warn(
                    f"sample_weighted = True. During the test split, groups with more samples will be prioritized",
                UserWarning,
            )

    def _iter_indices(self, X: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series], groups: Union[List, np.ndarray, pd.Series]):
        
        if y is None:
            raise ValueError(
                "StratifiedGroupShuffleSplit requires 'y' for stratification."
            )

        if groups is None:
            raise ValueError(
                "StratifiedGroupShuffleSplit requires 'groups' to be defined."
            )
        n_samples = _num_samples(X)

        if isinstance(self.test_size, float):
            n_test = int(self.test_size * n_samples)
        else:
            n_test = int(self.test_size)

        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        group_counts = np.bincount(groups)
        self._check_split_viability(n_test, unique_groups, group_counts)
        n_groups = len(unique_groups)
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        overall_class_counts = np.bincount(y_indices, minlength=n_classes)

        group_info = defaultdict(
            lambda: {
                "class_counts": np.zeros(n_classes, dtype=int),
                "indices": [],
                "size": 0,
            }
        )
        for i, group_idx in enumerate(group_indices):
            class_idx = y_indices[i]
            group_info[group_idx]["class_counts"][class_idx] += 1
            group_info[group_idx]["indices"].append(i)
        for i in range(n_groups):
            group_info[i]["size"] = len(group_info[i]["indices"])

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            available_groups = list(range(n_groups))
            test_groups = []

            current_test_size = 0
            current_test_counts = np.zeros(n_classes, dtype=int)

            # Phase 1: Greedily add only "safe" groups that do not exceed n_test
            while available_groups:
                safe_candidates = []
                for group_idx in available_groups:
                    group_data = group_info[group_idx]
                    if current_test_size + group_data["size"] <= n_test:
                        prospective_counts = (
                            current_test_counts + group_data["class_counts"]
                        )
                        prospective_size = current_test_size + group_data["size"]
                        ideal_counts = overall_class_counts * (
                            prospective_size / n_samples
                        )
                        error = np.sum((prospective_counts - ideal_counts) ** 2)
                        safe_candidates.append({"error": error, "id": group_idx})

                if not safe_candidates:
                    # No more groups can be added without overshooting
                    break

                safe_candidates.sort(key=lambda x: x["error"])
                pool_size = min(5, len(safe_candidates))
                candidate_pool = [cand["id"] for cand in safe_candidates[:pool_size]]
                if self.sample_weighted:
                    weights = [group_info[group_idx]["size"] for group_idx in candidate_pool]
                    best_group = rng.choice(candidate_pool, p=weights)
                else:
                    best_group = rng.choice(candidate_pool)

                test_groups.append(best_group)
                available_groups.remove(best_group)
                group_data = group_info[best_group]
                current_test_counts += group_data["class_counts"]
                current_test_size += group_data["size"]

            # Phase 2: Decide if a single overshoot is better than the current undershoot
            if available_groups and current_test_size < n_test:
                overshoot_candidates = []
                for group_idx in available_groups:
                    group_data = group_info[group_idx]
                    prospective_size = current_test_size + group_data["size"]
                    # We only care about the size difference now
                    overshoot_candidates.append(
                        {"id": group_idx, "size": prospective_size}
                    )

                if overshoot_candidates:
                    # Find the group that causes the smallest overshoot
                    overshoot_candidates.sort(key=lambda x: x["size"])
                    best_overshoot_group = overshoot_candidates[0]

                    undershoot_error = n_test - current_test_size
                    overshoot_error = best_overshoot_group["size"] - n_test

                    valid_overshoot_candidates = []
                    for cand in overshoot_candidates:
                        overshoot_error = cand["size"] - n_test
                        if overshoot_error < undershoot_error:
                            valid_overshoot_candidates.append(cand["id"])

                        if valid_overshoot_candidates:
                            # Randomly choose from the valid overshooting groups
                            best_overshoot_group_id = rng.choice(valid_overshoot_candidates)
                            test_groups.append(best_overshoot_group_id)

            test_indices = (
                np.concatenate([group_info[g_idx]["indices"] for g_idx in test_groups])
                if test_groups
                else []
            )
            if len(test_indices) == 0:
                raise RuntimeError(f"Given the dataset, no train/test split could be found. Try increasing test_size")
            all_indices = np.arange(n_samples)
            train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
            
            if isinstance(self.test_size, float):
                
                requested_test_size_ratio = self.test_size
            else:
                requested_test_size_ratio = self.test_size / n_samples
                
            test_size_error = np.abs(len(test_indices)/n_samples - requested_test_size_ratio) 
            
            if not self.suppress_warnings:
                if test_size_error > 0.05: # 5% deviation
                    warnings.warn(f"Requested and calculated test sizes differ by {test_size_error*100:.2f}%")

            yield train_indices, test_indices

    def split(self, X, y, groups=None):
        """Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), optional
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : array-like of shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Each group will be kept together in either the
            train set or the test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        yield from self._iter_indices(X, y, groups)

    def get_n_splits(self):
        return self.n_splits

    def _check_split_viability(self, n_test, unique_groups, group_counts):
        too_large_groups = {}
        n_groups = 0
        for group_id, group_count in zip(unique_groups, group_counts):
            if group_count >= n_test:
                n_groups += 1
                too_large_groups[group_id] = group_count
        if len(too_large_groups) > 0 and not self.suppress_warnings and n_groups < len(unique_groups):
            warnings.warn(
                f'''
                          Some groups are too large for the test set and will never be present in the test set: {too_large_groups}.\n 
                          If you want a group to be able to be present in the test set, test_size >= group_size.
                          ''',
                UserWarning,
            )
        elif len(too_large_groups) > 0 and not self.suppress_warnings and n_groups == len(unique_groups):
            warnings.warn(
                '''
                         "Warning: All available groups are larger than the target test size. 
                         The algorithm will still try to select a group that overshoots the target, 
                         which may lead to a larger than requested test set, or an completely empty test set."
                          ''',
                UserWarning,
            )
            

def train_test_group_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """Split arrays or matrices into random train and test subsets, while respecting group boundaries.

    Quick utility that wraps input validation and a Group-aware ShuffleSplit
    into a single call for splitting (and optionally subsampling) data in a
    one-liner.

    The last passed array is assumed to be the 'groups' array.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes. The last array must be the `groups`
        array.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. For group-based
        splitting, shuffling is always performed on the groups. If shuffle=False,
        a ValueError will be raised.

    stratify : array-like or bool, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels. If True, it will use the second to last array as
        stratification labels.
        Read more in the :ref:`User Guide <stratification>`.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    n_arrays = len(arrays)
    if n_arrays < 2:
        raise ValueError(
            "At least two arrays are required as input (e.g., X, groups)."
        )

    arrays = indexable(*arrays)
    groups = arrays[-1]

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if not shuffle:
        raise ValueError(
            "shuffle=False is not supported for train_test_group_split. "
            "Group-based splitting always shuffles the groups."
        )

    y_for_split = None
    if stratify is not None:
        if isinstance(stratify, bool):
            if stratify:  # stratify=True
                if n_arrays < 3:
                    raise ValueError(
                        "When stratify=True, at least three arrays are required as input (e.g., X, y, groups)."
                    )
                y_for_split = arrays[-2]
                CVClass = StratifiedGroupShuffleSplit
            else:  # stratify=False
                CVClass = GroupShuffleSplit
        else:  # stratify is an array
            y_for_split = stratify
            CVClass = StratifiedGroupShuffleSplit
    else:  # stratify is None
        CVClass = GroupShuffleSplit

    cv = CVClass(n_splits=1, test_size=n_test, train_size=n_train, random_state=random_state)

    train, test = next(cv.split(X=arrays[0], y=y_for_split, groups=groups))

    train, test = ensure_common_namespace_device(arrays[0], train, test)

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


class GroupSplitCV:
    """Cross-validator that performs group-aware splits.

    This cross-validator is a wrapper around StratifiedGroupShuffleSplit and
    GroupShuffleSplit to be used in scikit-learn's GridSearchCV and other
    similar utilities.

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    stratify : bool, default=False
        Whether to perform stratified sampling. If True, the `y` parameter
        in the `split` method is used for stratification.
    """
    def __init__(self, n_splits=5, *, test_size=0.2, train_size=None, random_state=None, stratify=False):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
            Stratification is done based on the y labels if `stratify=True`.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if self.stratify:
            if y is None:
                raise ValueError("The 'y' parameter should not be None when stratify=True.")
            cv = StratifiedGroupShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                train_size=self.train_size,
                random_state=self.random_state,
            )
            yield from cv.split(X, y, groups=groups)
        else:
            cv = GroupShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.test_size,
                train_size=self.train_size,
                random_state=self.random_state,
            )
            yield from cv.split(X, y=y, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits