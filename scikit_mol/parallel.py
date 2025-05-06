from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs


def parallelized_with_batches(
    fn: Callable[[Sequence[Any]], Any],
    inputs_list: Union[np.ndarray, pd.DataFrame],
    n_jobs: Optional[int] = None,
    **job_kwargs: Any,
) -> Sequence[Optional[Any]]:
    """
    Execute a function in parallel with batches of inputs.
    Parameters
    ----------
    fn : Callable[[Sequence[Any]], Any]
        The function to be executed in parallel. It should accept a sequence of inputs.
    inputs_list : np.ndarray
        The list of inputs to be processed in parallel.
    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel. If `None`, it will be determined automatically.
    **job_kwargs : Any
        Additional keyword arguments to pass to the `joblib.Parallel` object.
    Returns
    -------
    Sequence[Optional[Any]]
        A sequence of results from the function executed in parallel.
    """
    n_jobs = effective_n_jobs(n_jobs)
    if n_jobs > len(inputs_list):
        n_jobs = len(inputs_list)
    pool = Parallel(n_jobs=n_jobs, **job_kwargs)

    if isinstance(inputs_list, (pd.DataFrame, pd.Series)):
        indexes = np.array_split(range(len(inputs_list)), n_jobs)
        input_chunks = [inputs_list.iloc[idx] for idx in indexes]
    else:
        input_chunks = np.array_split(inputs_list, n_jobs)

    results = pool(delayed(fn)(chunk) for chunk in input_chunks)
    return results
