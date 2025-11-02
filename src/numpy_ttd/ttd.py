from collections.abc import Callable, Iterable
from types import EllipsisType
from typing import Self, overload

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import NDArray

HANDLED_FUNCTIONS: dict[Callable, Callable] = {}


def implements(np_function: Callable) -> Callable:
    """Register an __array_function__ implementation for TTD objects."""

    def decorator(func: Callable) -> Callable:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class TTD(NDArrayOperatorsMixin):
    """
    Class for storing TTD encoded data.

    The class on the outside behaves like a Numpy ndarray but internally it
    stores the data in a compressed form using a TTD (tensor train
    decomposition). It also tries to perform all operations using this form but
    falls back on expanding to full ndarray if necessary.
    """

    data: list[NDArray]

    def __init__(self, data: Iterable[NDArray] | None = None):
        """
        Create a new empty TTD object.

        Parameters
        ----------
        data : Iterable[NDArray], optional
            The data to store in the TTD object. If provided, the caller must
            ensure that the data represents a valid TTD. Defaults to an empty
            decomposition.

        Returns
        -------
            TTD object

        """
        super().__init__()
        self.data = list(data) if data is not None else []

    @staticmethod
    def from_ndarray(array: NDArray) -> Self:
        """Compress an ndarray into a TTD object."""
        # TODO: implement compression
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD({self.data})"

    def __str__(self) -> str:
        """Return a string representation of the TTD object."""
        return f"TTD({self.data})"

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray:
        """
        Expand a TTD object into a full ndarray.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to use for the resulting NumPy array. By default,
            the dtype is inferred from the data.

        copy : bool or None, optional
            See :func:`numpy.asarray`.

        Returns
        -------
        numpy.ndarray
            The values in the series converted to a :class:`numpy.ndarray`
            with the specified `dtype`.

        """
        raise NotImplementedError

    def __array_ufunc__(
        self, ufunc: Callable, method: str, *args: tuple, **kwargs: dict
    ) -> Self:
        """
        Apply a numpy ufunc to a TTD object.

        Parameters
        ----------
        ufunc : Callable
            The numpy ufunc to apply.
        method : str
            The method to use for the ufunc.
        *args : list
            The inputs to the ufunc.
        **kwargs : dict
            The keyword arguments to the ufunc.

        Returns
        -------
        TTD
            The result of the ufunc applied to the TTD object.

        """
        if method == "__call__":
            # TODO: implement ufuncs
            return self.__class__()

        # fallback to operating on __array__
        return NotImplemented

    def __array_function__(
        self, func: Callable, types: tuple[type], args: tuple, kwargs: dict
    ):
        """
        Call a numpy method on a TTD object.

        Parameters
        ----------
        func : Callable
            The numpy method to call.
        types : tuple[type]
            The types of the arguments.
        args : tuple
            The arguments to the numpy method.
        kwargs : dict
            The keyword arguments to the numpy method.

        Returns
        -------
        TTD
            The result of the numpy method applied to the TTD object.

        """
        handler = HANDLED_FUNCTIONS.get(func)

        if handler is None:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle TTD objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return handler(*args, **kwargs)

    @implements(np.sum)
    def sum(self: Self) -> float:
        """Sum the elements on the TTD object."""
        # TODO: implement
        raise NotImplementedError

    # TODO: implement other numpy functions

    def _to_raw(self) -> list[NDArray]:
        """Retrieve the internal representation as a list of ndarrays."""
        return self.data


    @overload
    def __getitem__(self, key: tuple[int, ...]) -> NDArray: ...
    @overload
    def __getitem__(self, key: EllipsisType) -> Self: ...

    def __getitem__(self, key: EllipsisType | tuple[int, ...]) -> Self | NDArray:
        """Get a single values from the TTD object."""
        if key == Ellipsis:
            return self.__class__(a[...] for a in self.data)

        raise NotImplementedError
