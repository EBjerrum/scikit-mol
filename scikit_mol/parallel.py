from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np
from joblib import Parallel, delayed


def parallelized_with_batches(
    fn: Callable,
    inputs_list: np.ndarray,
    n_jobs: Optional[int] = None,
    **job_kwargs: Any,
) -> Sequence[Optional[Any]]:
    pool = Parallel(n_jobs=n_jobs, **job_kwargs)
    n_jobs = pool._effective_n_jobs()
    if n_jobs > len(inputs_list):
        n_jobs = len(inputs_list)
        pool = Parallel(n_jobs=n_jobs, **job_kwargs)

    input_chunks = np.array_split(inputs_list, n_jobs)
    results = pool(delayed(fn)(chunk) for chunk in input_chunks)
    return results
