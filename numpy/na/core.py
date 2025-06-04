import numpy as np
import functools
from typing import Dict, List, Union, Optional, Tuple, Any
import warnings

__all__ = [
    "NamedArray",
    "named_array",
    "zeros_named",
    "ones_named",
    "random_named",
]

class NamedArray:
    """
    A lightweight wrapper around numpy arrays that adds named dimension support.
    
    This class implements NumPy's interoperability protocols (__array__, __array_interface__)
    to ensure seamless integration with existing NumPy functions while adding named
    dimension functionality for improved code readability and maintainability.
    """
    
    def __init__(self, array: np.ndarray, dim_names: Optional[List[str]] = None):
        """
        Initialize a NamedArray.
        
        Parameters
        ----------
        array : np.ndarray
            The underlying numpy array
        dim_names : list of str, optional
            Names for each dimension. If None, dimensions will be unnamed.
            Length must match array.ndim.
        """
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        
        self._array = array
        
        if dim_names is None:
            self._dim_names = [f"dim_{i}" for i in range(array.ndim)]
        else:
            if len(dim_names) != array.ndim:
                raise ValueError(f"Number of dimension names ({len(dim_names)}) "
                               f"must match array dimensions ({array.ndim})")
            # Check for duplicate names
            if len(set(dim_names)) != len(dim_names):
                raise ValueError("Dimension names must be unique")
            self._dim_names = list(dim_names)
    
    @property
    def dim_names(self) -> List[str]:
        """Get the dimension names."""
        return self._dim_names.copy()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the underlying array."""
        return self._array.shape
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return self._array.ndim
    
    @property
    def dtype(self):
        """Get the data type."""
        return self._array.dtype
    
    @property
    def size(self) -> int:
        """Get the total number of elements."""
        return self._array.size
    
    def __array__(self, dtype=None):
        """NumPy array interface for seamless integration."""
        if dtype is not None:
            return self._array.astype(dtype)
        return self._array
    
    @property
    def __array_interface__(self):
        """NumPy array interface dictionary."""
        return self._array.__array_interface__
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle universal functions while preserving named dimensions."""
        # Convert NamedArray inputs to regular arrays
        args = []
        named_input = None
        
        for input_ in inputs:
            if isinstance(input_, NamedArray):
                args.append(input_._array)
                if named_input is None:
                    named_input = input_
            else:
                args.append(input_)
        
        # Call the ufunc
        result = getattr(ufunc, method)(*args, **kwargs)
        
        # Wrap result if it's an array and we have named dimensions
        if isinstance(result, np.ndarray) and named_input is not None:
            # For reductions, we need to handle dimension changes
            if result.ndim < named_input.ndim:
                # This is likely a reduction - we'll need to infer which dims were reduced
                # For now, just create generic names
                new_dim_names = [f"dim_{i}" for i in range(result.ndim)]
            else:
                new_dim_names = named_input._dim_names
            
            return NamedArray(result, new_dim_names)
        
        return result
    
    def __getitem__(self, key):
        """Handle slicing while preserving appropriate dimension names."""
        result = self._array[key]
        
        if not isinstance(result, np.ndarray):
            return result
        
        # Handle dimension name preservation for slicing
        if isinstance(key, tuple):
            # Complex slicing - need to track which dimensions remain
            new_dim_names = []
            for i, (slice_obj, dim_name) in enumerate(zip(key, self._dim_names)):
                if not isinstance(slice_obj, int):  # Keep dimension if not integer indexing
                    new_dim_names.append(dim_name)
            # Add remaining dimensions not covered by the key
            remaining_dims = len(self._dim_names) - len(key)
            new_dim_names.extend(self._dim_names[-remaining_dims:] if remaining_dims > 0 else [])
        else:
            # Simple slicing
            if isinstance(key, int):
                new_dim_names = self._dim_names[1:]  # Remove first dimension
            else:
                new_dim_names = self._dim_names  # Keep all dimensions
        
        return NamedArray(result, new_dim_names[:result.ndim])
    
    def __repr__(self):
        """String representation showing both array and dimension names."""
        return f"NamedArray({self._array}, dim_names={self._dim_names})"
    
    def __str__(self):
        """String representation."""
        return f"NamedArray with dimensions {dict(zip(self._dim_names, self.shape))}\n{self._array}"
    
    def reshape_named(self, shape_dict: Dict[str, Union[int, Tuple[str, ...]]]) -> 'NamedArray':
        """
        Reshape array using named dimensions.
        
        Parameters
        ----------
        shape_dict : dict
            Dictionary mapping dimension names to new shapes. Can specify:
            - Simple reshape: {'height': 10, 'width': 20}
            - Dimension splitting: {'spatial': ('height', 'width')}
            - Dimension merging: {('batch', 'time'): 'sequence'}
        
        Returns
        -------
        NamedArray
            Reshaped array with updated dimension names
        """
        # Parse the shape dictionary to determine new shape and names
        new_shape = []
        new_dim_names = []
        old_dim_sizes = dict(zip(self._dim_names, self.shape))
        
        for key, value in shape_dict.items():
            if isinstance(key, str):
                # Simple dimension or splitting
                if isinstance(value, (int, type(None))):
                    # Simple reshape
                    new_shape.append(value if value != -1 else -1)
                    new_dim_names.append(key)
                elif isinstance(value, tuple):
                    # Dimension splitting - need to compute individual sizes
                    if key not in old_dim_sizes:
                        raise ValueError(f"Dimension '{key}' not found in array")
                    
                    total_size = old_dim_sizes[key]
                    # For now, require explicit sizes for splits
                    raise NotImplementedError("Dimension splitting requires explicit size specification")
            elif isinstance(key, tuple):
                # Dimension merging
                if isinstance(value, str):
                    # Merge dimensions
                    merge_size = 1
                    for dim in key:
                        if dim not in old_dim_sizes:
                            raise ValueError(f"Dimension '{dim}' not found in array")
                        merge_size *= old_dim_sizes[dim]
                    new_shape.append(merge_size)
                    new_dim_names.append(value)
        
        # Handle remaining dimensions not specified in shape_dict
        for dim_name in self._dim_names:
            if dim_name not in [k for k in shape_dict.keys() if isinstance(k, str)]:
                # Check if this dimension is part of a merge operation
                is_merged = any(dim_name in k for k in shape_dict.keys() if isinstance(k, tuple))
                if not is_merged:
                    new_shape.append(old_dim_sizes[dim_name])
                    new_dim_names.append(dim_name)
        
        # Perform the reshape
        try:
            reshaped_array = self._array.reshape(new_shape)
            return NamedArray(reshaped_array, new_dim_names)
        except ValueError as e:
            raise ValueError(f"Cannot reshape array: {e}")
    
    def transpose_named(self, axes_names: List[str]) -> 'NamedArray':
        """
        Transpose array using named dimensions.
        
        Parameters
        ----------
        axes_names : list of str
            List of dimension names in the desired order
        
        Returns
        -------
        NamedArray
            Transposed array with reordered dimensions
        """
        if len(axes_names) != len(self._dim_names):
            raise ValueError(f"Number of axes ({len(axes_names)}) must match "
                           f"array dimensions ({len(self._dim_names)})")
        
        # Check that all provided names exist
        for name in axes_names:
            if name not in self._dim_names:
                raise ValueError(f"Dimension name '{name}' not found in array")
        
        # Check for duplicates
        if len(set(axes_names)) != len(axes_names):
            raise ValueError("Axis names must be unique")
        
        # Create the axes permutation
        axes = [self._dim_names.index(name) for name in axes_names]
        
        # Perform the transpose
        transposed_array = self._array.transpose(axes)
        return NamedArray(transposed_array, axes_names)
    
    def sum_named(self, axis: Union[str, List[str]], keepdims: bool = False) -> 'NamedArray':
        """
        Sum along named dimensions.
        
        Parameters
        ----------
        axis : str or list of str
            Dimension name(s) to sum along
        keepdims : bool, default False
            Whether to keep reduced dimensions
        
        Returns
        -------
        NamedArray
            Array with specified dimensions summed
        """
        if isinstance(axis, str):
            axis = [axis]
        
        # Convert names to indices
        axis_indices = []
        for name in axis:
            if name not in self._dim_names:
                raise ValueError(f"Dimension name '{name}' not found in array")
            axis_indices.append(self._dim_names.index(name))
        
        # Perform the sum
        result = np.sum(self._array, axis=tuple(axis_indices), keepdims=keepdims)
        
        # Update dimension names
        if keepdims:
            new_dim_names = self._dim_names.copy()
        else:
            new_dim_names = [name for name in self._dim_names if name not in axis]
        
        return NamedArray(result, new_dim_names)
    
    def mean_named(self, axis: Union[str, List[str]], keepdims: bool = False) -> 'NamedArray':
        """Mean along named dimensions."""
        if isinstance(axis, str):
            axis = [axis]
        
        axis_indices = []
        for name in axis:
            if name not in self._dim_names:
                raise ValueError(f"Dimension name '{name}' not found in array")
            axis_indices.append(self._dim_names.index(name))
        
        result = np.mean(self._array, axis=tuple(axis_indices), keepdims=keepdims)
        
        if keepdims:
            new_dim_names = self._dim_names.copy()
        else:
            new_dim_names = [name for name in self._dim_names if name not in axis]
        
        return NamedArray(result, new_dim_names)


# Convenience functions
def named_array(array, dim_names=None):
    """Create a NamedArray from a regular array."""
    return NamedArray(array, dim_names)


def zeros_named(shape_dict: Dict[str, int], dtype=np.float64) -> NamedArray:
    """Create a NamedArray of zeros with named dimensions."""
    dim_names = list(shape_dict.keys())
    shape = list(shape_dict.values())
    array = np.zeros(shape, dtype=dtype)
    return NamedArray(array, dim_names)


def ones_named(shape_dict: Dict[str, int], dtype=np.float64) -> NamedArray:
    """Create a NamedArray of ones with named dimensions."""
    dim_names = list(shape_dict.keys())
    shape = list(shape_dict.values())
    array = np.ones(shape, dtype=dtype)
    return NamedArray(array, dim_names)


def random_named(shape_dict: Dict[str, int]) -> NamedArray:
    """Create a NamedArray of random values with named dimensions."""
    dim_names = list(shape_dict.keys())
    shape = list(shape_dict.values())
    array = np.random.random(shape)
    return NamedArray(array, dim_names)