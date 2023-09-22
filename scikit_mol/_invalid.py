from abc import ABC
from typing import Any, Callable, NamedTuple, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

_T = TypeVar("_T")
_U = TypeVar("_U")


class InvalidInstance(NamedTuple):
    pipeline_step: str
    error: str


class NumpyArrayWithInvalidInstances:
    is_valid_array: npt.NDArray[np.bool_]
    invalid_list: list[InvalidInstance]
    value_array: npt.NDArray[Any]

    def __init__(self, array_list: list[npt.NDArray[Any] | InvalidInstance]):
        self.is_valid_array = get_is_valid_array(array_list)
        valid_vector_list = filter_by_list(array_list, self.is_valid_array)
        self.value_array = np.vstack(valid_vector_list)
        self.invalid_list = filter_by_list(array_list, ~self.is_valid_array)

    def __getitem__(self, item: int) -> npt.NDArray[Any] | InvalidInstance:
        n_invalids_prior = sum(~self.is_valid_array[:item - 1])
        if self.is_valid_array[item]:
            return self.value_array[item - n_invalids_prior]
        return self.invalid_list[n_invalids_prior + 1]

    def __setitem__(self, key: int, value: npt.NDArray[Any] | InvalidInstance) -> None:
        n_invalids_prior = sum(~self.is_valid_array[:key - 1])
        if isinstance(value, InvalidInstance):
            if self.is_valid_array[key]:
                self.value_array = np.delete(self.value_array, key - n_invalids_prior)
                self.is_valid_array[key] = False
                self.invalid_list.insert(n_invalids_prior + 1, value)
            else:
                self.invalid_list[n_invalids_prior + 1] = value
        else:
            if self.is_valid_array[key]:
                self.value_array[key - n_invalids_prior] = value
            else:
                self.value_array = np.insert(self.value_array, key - n_invalids_prior, value)
                del(self.invalid_list[n_invalids_prior + 1])
                self.is_valid_array[key] = True

    def array_filled_with(self, fill_value) -> npt.NDArray[Any]:
        out = np.full((len(self.is_valid_array), self.value_array.shape[1]), fill_value)
        out[self.is_valid_array] = self.value_array
        return out


def batch_update_sequence(
        old_list: list[npt.NDArray[Any] | InvalidInstance] | NumpyArrayWithInvalidInstances,
        new_values: list[Any],
        value_indices: npt.NDArray[np.int_],
   ):
    old_list = list(old_list)  # Make shallow copy of list to avoid inplace changes.
    for new_value, idx in zip(new_values, value_indices, strict=True):
        old_list[idx] = new_value
    return old_list


def filter_by_list(item_list, is_valid_array: npt.NDArray[np.bool_]):
    if isinstance(item_list, np.ndarray):
        return item_list[is_valid_array]

    item_list_new = []
    for item, is_valid in zip(item_list, is_valid_array):
        if is_valid:
            item_list_new.append(item)
    return item_list_new


# Callable[[Sequence[Any], Sequence[Any], dict[str, Any]], Sequence[Any]]
# ) -> Callable[[Sequence[Any], Sequence[Any], dict[str, Any]], npt.NDArray[Any]]
def rdkit_error_handling(func):
    def wrapper(obj, *args, **kwargs):
        x = args[0]
        if isinstance(x, NumpyArrayWithInvalidInstances):
            is_valid_array = x.is_valid_array
            x_sub = x.value_array
        else:
            is_valid_array = get_is_valid_array(x)
            x_sub = filter_by_list(x, is_valid_array)
        if len(args) > 1:
            y = args[1]
            if y is not None:
                y_sub = filter_by_list(y, is_valid_array)
            else:
                y_sub = None
            x_new = func(obj, x_sub, y_sub, **kwargs)
        else:
            x_new = func(obj, x_sub, **kwargs)

        if x_new is None:  # fit may not return anything
            return None
        new_pos = np.where(is_valid_array)[0]
        if isinstance(x_new, np.ndarray) and isinstance(x, NumpyArrayWithInvalidInstances):
            if x_new.shape[0] != x.value_array.shape[0]:
                raise AssertionError("Numer of rows is not as expected.")
            x.value_array = x_new
            return x
        if isinstance(x, (list, NumpyArrayWithInvalidInstances)):
            x_list = batch_update_sequence(x, x_new, new_pos)
        else:
            x_array = np.array(x)
            x_array[is_valid_array] = x_new
            x_list = list(x_array)
        if isinstance(x_new, NumpyArrayWithInvalidInstances):
            return NumpyArrayWithInvalidInstances(x_list)
        return x_list
    return wrapper


def filter_rows(
    X: Sequence[_T], y: Sequence[_U]
) -> tuple[Sequence[_T], Sequence[_U]]:
    is_valid_array = get_is_valid_array(X)
    x_new = filter_by_list(X, is_valid_array)
    y_new = filter_by_list(y, is_valid_array)
    return x_new, y_new


def get_is_valid_array(item_list: Sequence[Any]) -> npt.NDArray[np.bool_]:
    is_valid_list = []
    for i, item in enumerate(item_list):
        if not isinstance(item, InvalidInstance):
            is_valid_list.append(True)
        else:
            is_valid_list.append(False)
    return np.array(is_valid_list, dtype=bool)


