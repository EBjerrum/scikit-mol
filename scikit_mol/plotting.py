import os
import platform
import re
import subprocess
import time
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ParallelTester:
    """
    A class to test the performance of a transformer on a set of molecules using parallel processing.
    Parameters
    ----------
    transformer : object
        The transformer object that has a `transform` method to apply to the molecules.
    mols : list
        A list of molecules to be transformed.
    n_mols : tuple, optional
        A tuple of integers specifying the number of molecules to test with. Default is (10, 100, 1000, 10000, 100000).
    n_jobs : tuple, optional
        A tuple of integers specifying the number of parallel jobs to test with. Default is (1, 2, 4, 8).
    backend : str, optional
        The parallel backend to use. Default is 'loky'.
    Attributes
    ----------
    mols : list
        The list of molecules to be transformed.
    n_mols : list
        The sorted list of number of molecules to test with.
    n_jobs : list
        The sorted list of number of parallel jobs to test with.
    transformer : object
        The transformer object.
    backend : str
        The parallel backend to use.
    Methods
    -------
    _test_single(mols, n_jobs)
        Tests the transformer on a subset of molecules with a specified number of parallel jobs.
    test()
        Tests the transformer on various subsets of molecules with different numbers of parallel jobs and returns the results as a DataFrame.
    """

    def __init__(
        self,
        transformer,
        mols,
        n_mols=(10, 100, 100, 1000, 10000, 100000),
        n_jobs=(1, 2, 4, 8),
        backend="loky",
    ):
        self.mols = mols
        n_mols = sorted(n_mols)
        if max(n_mols) > len(mols):
            raise ValueError(
                f"Maximum number of molecules {max(n_mols)} is greater than the number of molecules {len(mols)}"
            )
        self.n_mols = n_mols
        n_jobs = sorted(n_jobs)
        if max(n_jobs) > os.cpu_count():
            raise ValueError(
                f"Maximum number of jobs {max(n_jobs)} is greater than the number of CPUs {os.cpu_count()}"
            )
        self.n_jobs = n_jobs
        self.transformer = transformer
        self.backend = backend

    def _test_single(self, mols, n_jobs):
        start = time.perf_counter()
        with joblib.parallel_backend(self.backend, n_jobs=n_jobs):
            self.transformer.transform(mols)
        return time.perf_counter() - start

    def test(self):
        results = pd.DataFrame(columns=self.n_mols, index=self.n_jobs)
        for n_mol in self.n_mols:
            for n_job in self.n_jobs:
                results.at[n_job, n_mol] = self._test_single(self.mols[:n_mol], n_job)
        return results


def get_processor_name() -> str:
    """
    Retrieves the name of the processor on the current system.

    Returns
    -------
    str
        The name of the processor. Returns an empty string if the processor name cannot be determined.

    Notes
    -----
    - On Windows, it uses `platform.processor()`.
    - On macOS (Darwin), it uses the `sysctl` command to get the CPU brand string.
    - On Linux, it reads `/proc/cpuinfo` to find the model name.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()  # noqa: S603
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()  # noqa: S602
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(
                    pattern=".*model name.*:", repl="", string=line, count=1
                ).strip()
    return ""


def plot_heatmap(df: pd.DataFrame, name: Optional[str] = None, normalize: bool = True):
    """
    Plots a heatmap of the given DataFrame from the parallel tester.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    name : str, optional
        The name to be used in the title of the plot. Default is None.
    normalize : bool, optional
        If True, normalize the DataFrame by the first row. Default is True.

    Returns
    -------
        The heatmap plot.

    Notes
    -----
    The function normalizes the DataFrame by dividing by the first row if `normalize` is True.
    The colormap used is "PiYG_r". The title of the plot includes the maximum single-threaded speed
    and the CPU name.
    """
    df = df.astype(float)
    max_speed = (df.columns / df.loc[1]).max()
    v_min, v_max = None, None
    if normalize:
        df = df / df.loc[1]
        cmap = sns.color_palette("PiYG_r", as_cmap=True)
        v_min = 0.0
        v_max = 2.0
    else:
        cmap = sns.color_palette("PiYG_r", as_cmap=True)
    plt.figure(figsize=(8, 6))
    g = sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, vmin=v_min, vmax=v_max)
    title = ""
    if name:
        title = name + "\n"
    title += f"Max single-threaded speed {max_speed:.0f} mols/s"
    title += "\nCPU: " + get_processor_name()
    plt.title(title)
    plt.xlabel("Number of mols")
    plt.ylabel("Number of jobs")
    return g
