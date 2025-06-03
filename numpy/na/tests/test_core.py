"""Tests suite for NamedArray & named operations.

:author: Test Author
:contact: test_at_example_dot_com
"""
__author__ = "Test Author"

import copy
import itertools
import operator
import pickle
import sys
import textwrap
import warnings
import time
from functools import reduce

import pytest

import numpy as np
import numpy._core.fromnumeric as fromnumeric
import numpy._core.umath as umath
from numpy import ndarray
from numpy._utils import asbytes
from numpy.exceptions import AxisError
from numpy.testing import (
    IS_WASM,
    assert_raises,
    assert_warns,
    suppress_warnings,
    temppath,
)
from numpy.testing._private.utils import requires_memory

# Import NamedArray components
from numpy.na import (
    NamedArray,
    named_array,
    zeros_named,
    ones_named,
    random_named,
)

# Test utilities
from numpy.testing import (
    assert_equal,
    assert_array_equal,
    assert_almost_equal,
    assert_,
)

pi = np.pi

# For parametrized numeric testing
num_dts = [np.dtype(dt_) for dt_ in '?bhilqBHILQefdgFD']
num_ids = [dt_.char for dt_ in num_dts]


class TestNamedArray:
    # Base test class for NamedArrays.

    def setup_method(self):
        # Base data definition.
        np.random.seed(42)  # For reproducible tests
        x = np.array([1., 1., 1., -2., pi / 2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        z = np.array([-.5, 0., .5, .8])
        
        # 3D test arrays
        arr3d = np.random.random((3, 4, 5))
        arr2d = np.random.random((4, 6))
        
        # Named arrays
        x_named = NamedArray(x, ['time'])
        y_named = NamedArray(y, ['time'])
        z_named = NamedArray(z, ['batch'])
        arr3d_named = NamedArray(arr3d, ['batch', 'height', 'width'])
        arr2d_named = NamedArray(arr2d, ['samples', 'features'])
        
        # Default named arrays
        arr3d_default = NamedArray(arr3d)
        
        self.d = (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
                 arr3d_named, arr2d_named, arr3d_default)

    def test_basicattributes(self):
        # Tests some basic array attributes.
        a = NamedArray([1, 3, 2], ['time'])
        b = NamedArray([[1, 2], [3, 4]], ['batch', 'features'])
        assert_equal(a.ndim, 1)
        assert_equal(b.ndim, 2)
        assert_equal(a.size, 3)
        assert_equal(b.size, 4)
        assert_equal(a.shape, (3,))
        assert_equal(b.shape, (2, 2))
        assert_equal(a.dim_names, ['time'])
        assert_equal(b.dim_names, ['batch', 'features'])

    def test_basic0d(self):
        # Checks 0-dimensional arrays
        x = NamedArray(5, [])
        assert_equal(x.shape, ())
        assert_equal(x.dim_names, [])
        assert_equal(x.ndim, 0)
        assert_equal(x.size, 1)

    def test_basic1d(self):
        # Test of basic array creation and properties in 1 dimension.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        assert_(not isinstance(x, NamedArray))
        assert_(isinstance(x_named, NamedArray))
        assert_equal(x_named.shape, x.shape)
        assert_equal(x_named.dtype, x.dtype)
        assert_equal(x_named.dim_names, ['time'])
        assert_equal(x_named.size, x.size)
        assert_array_equal(x_named._array, x)
        assert_array_equal(np.array(x_named), x)

    def test_basic2d(self):
        # Test of basic array creation and properties in 2 dimensions.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        assert_equal(arr2d_named.shape, arr2d.shape)
        assert_equal(arr2d_named.ndim, 2)
        assert_equal(arr2d_named.dim_names, ['samples', 'features'])
        assert_array_equal(arr2d_named._array, arr2d)

    def test_basic3d(self):
        # Test of basic array creation and properties in 3 dimensions.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        assert_equal(arr3d_named.shape, (3, 4, 5))
        assert_equal(arr3d_named.ndim, 3)
        assert_equal(arr3d_named.dim_names, ['batch', 'height', 'width'])
        assert_array_equal(arr3d_named._array, arr3d)

    def test_creation_default_names(self):
        # Test creation with default dimension names.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        expected_names = ['dim_0', 'dim_1', 'dim_2']
        assert_equal(arr3d_default.dim_names, expected_names)
        assert_equal(arr3d_default.shape, arr3d.shape)

    def test_creation_errors(self):
        # Test invalid creation scenarios.
        arr = np.random.random((3, 4))
        
        # Wrong number of dimension names
        assert_raises(ValueError, NamedArray, arr, ['batch'])
        assert_raises(ValueError, NamedArray, arr, ['batch', 'height', 'width'])
        
        # Duplicate dimension names
        assert_raises(ValueError, NamedArray, arr, ['batch', 'batch'])

    def test_numpy_interoperability(self):
        # Test NumPy interoperability protocols.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        # Test __array__ method
        np_array = np.array(x_named)
        assert_array_equal(np_array, x)
        
        # Test with NumPy functions
        sum_result = np.sum(x_named)
        expected_sum = np.sum(x)
        assert_almost_equal(sum_result, expected_sum)
        
        # Test reshape with NumPy
        reshaped = np.reshape(arr2d_named, (24,))
        assert_equal(reshaped.shape, (24,))

    def test_transpose_named_valid(self):
        # Test valid named transpose operations.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        # Basic transpose
        transposed = arr3d_named.transpose_named(['width', 'batch', 'height'])
        assert_equal(transposed.shape, (5, 3, 4))
        assert_equal(transposed.dim_names, ['width', 'batch', 'height'])
        
        # Verify data integrity
        original_data = arr3d_named._array[1, 2, 3]
        transposed_data = transposed._array[3, 1, 2]
        assert_equal(original_data, transposed_data)
        
        # Single dimension transpose (should be identity)
        single_transposed = x_named.transpose_named(['time'])
        assert_equal(single_transposed.shape, x_named.shape)
        assert_equal(single_transposed.dim_names, x_named.dim_names)

    def test_transpose_named_errors(self):
        # Test invalid named transpose operations.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        # Wrong number of axes
        assert_raises(ValueError, arr3d_named.transpose_named, ['batch', 'height'])
        
        # Non-existent dimension name
        assert_raises(ValueError, arr3d_named.transpose_named, 
                     ['batch', 'height', 'invalid'])
        
        # Duplicate axes
        assert_raises(ValueError, arr3d_named.transpose_named, 
                     ['batch', 'batch', 'width'])

    def test_sum_named_single_axis(self):
        # Test sum along single named axis.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        # Sum along batch dimension
        sum_batch = arr3d_named.sum_named('batch')
        assert_equal(sum_batch.shape, (4, 5))
        assert_equal(sum_batch.dim_names, ['height', 'width'])
        
        expected_sum = np.sum(arr3d, axis=0)
        assert_array_equal(sum_batch._array, expected_sum)

    def test_sum_named_multiple_axes(self):
        # Test sum along multiple named axes.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        # Sum along multiple dimensions
        sum_multi = arr3d_named.sum_named(['batch', 'height'])
        assert_equal(sum_multi.shape, (5,))
        assert_equal(sum_multi.dim_names, ['width'])
        
        expected_sum = np.sum(arr3d, axis=(0, 1))
        assert_array_equal(sum_multi._array, expected_sum)

    def test_sum_named_keepdims(self):
        # Test sum with keepdims option.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        sum_keepdims = arr3d_named.sum_named('batch', keepdims=True)
        assert_equal(sum_keepdims.shape, (1, 4, 5))
        assert_equal(sum_keepdims.dim_names, ['batch', 'height', 'width'])

    def test_mean_named(self):
        # Test mean operation along named axes.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        mean_result = arr3d_named.mean_named('width')
        assert_equal(mean_result.shape, (3, 4))
        assert_equal(mean_result.dim_names, ['batch', 'height'])
        
        expected_mean = np.mean(arr3d, axis=2)
        assert_array_equal(mean_result._array, expected_mean)

    def test_reduction_errors(self):
        # Test invalid reduction operations.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        # Non-existent dimension
        assert_raises(ValueError, arr3d_named.sum_named, 'nonexistent')
        assert_raises(ValueError, arr3d_named.mean_named, 'invalid')

    def test_slicing_integer_indexing(self):
        # Test integer indexing (removes dimension).
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        subset = arr3d_named[0]
        assert_equal(subset.shape, (4, 5))
        assert_equal(subset.dim_names, ['height', 'width'])
        assert_(isinstance(subset, NamedArray))

    def test_slicing_slice_indexing(self):
        # Test slice indexing (keeps dimension).
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        subset_slice = arr3d_named[0:2]
        assert_equal(subset_slice.shape, (2, 4, 5))
        assert_equal(subset_slice.dim_names, ['batch', 'height', 'width'])

    def test_slicing_complex(self):
        # Test complex slicing operations.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        complex_subset = arr3d_named[0:2, :, 1:4]
        assert_equal(complex_subset.shape, (2, 4, 3))
        assert_equal(complex_subset.dim_names, ['batch', 'height', 'width'])

    def test_ufunc_addition(self):
        # Test universal function addition.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        # Addition with another NamedArray
        other_array = NamedArray(np.ones_like(arr3d), ['batch', 'height', 'width'])
        result = arr3d_named + other_array
        assert_(isinstance(result, NamedArray))
        assert_equal(result.dim_names, arr3d_named.dim_names)

    def test_ufunc_scalar_multiplication(self):
        # Test universal function with scalar.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        result_scalar = arr3d_named * 2
        assert_(isinstance(result_scalar, NamedArray))
        assert_array_equal(result_scalar._array, arr3d * 2)
        assert_equal(result_scalar.dim_names, arr3d_named.dim_names)

    def test_convenience_zeros_named(self):
        # Test zeros_named convenience function.
        zeros_arr = zeros_named({'batch': 10, 'features': 5})
        assert_equal(zeros_arr.shape, (10, 5))
        assert_equal(zeros_arr.dim_names, ['batch', 'features'])
        assert_array_equal(zeros_arr._array, np.zeros((10, 5)))

    def test_convenience_ones_named(self):
        # Test ones_named convenience function.
        ones_arr = ones_named({'time': 3, 'channels': 2}, dtype=np.int32)
        assert_equal(ones_arr.shape, (3, 2))
        assert_equal(ones_arr.dtype, np.int32)
        assert_array_equal(ones_arr._array, np.ones((3, 2), dtype=np.int32))

    def test_convenience_random_named(self):
        # Test random_named convenience function.
        random_arr = random_named({'height': 4, 'width': 4})
        assert_equal(random_arr.shape, (4, 4))
        assert_(0 <= random_arr._array.min() <= 1)
        assert_(0 <= random_arr._array.max() <= 1)

    def test_edge_case_single_dimension(self):
        # Test single dimension array operations.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        single_dim = NamedArray(np.array([1, 2, 3]), ['time'])
        transposed_single = single_dim.transpose_named(['time'])
        assert_equal(transposed_single.shape, (3,))
        assert_equal(transposed_single.dim_names, ['time'])

    def test_edge_case_large_array(self):
        # Test performance with large arrays.
        large_array = np.random.random((100, 100, 100))
        large_named = NamedArray(large_array, ['x', 'y', 'z'])
        transposed_large = large_named.transpose_named(['z', 'x', 'y'])
        assert_equal(transposed_large.shape, (100, 100, 100))
        assert_equal(transposed_large.dim_names, ['z', 'x', 'y'])

    def test_error_messages_informative(self):
        # Test that error messages are informative.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        try:
            arr3d_named.transpose_named(['batch', 'invalid'])
        except ValueError as e:
            assert_('invalid' in str(e))
        
        try:
            arr3d_named.sum_named('nonexistent')
        except ValueError as e:
            assert_('nonexistent' in str(e))

    def test_string_representations(self):
        # Test string representations of NamedArray.
        (x, y, z, arr3d, arr2d, x_named, y_named, z_named, 
         arr3d_named, arr2d_named, arr3d_default) = self.d
        
        repr_str = repr(arr3d_named)
        assert_('NamedArray' in repr_str)
        assert_('dim_names' in repr_str)
        
        str_str = str(arr3d_named)
        assert_('NamedArray' in str_str)
        assert_('batch' in str_str)


class TestNamedArrayPerformance:
    # Performance benchmarks for NamedArray operations.

    def setup_method(self):
        # Set up performance test fixtures.
        self.large_array = np.random.random((1000, 500, 100))
        self.dim_names = ['batch', 'features', 'time']
        self.named_large = NamedArray(self.large_array, self.dim_names)

    def benchmark_operation(self, operation_name, named_op, numpy_op, iterations=10):
        # Benchmark a named operation against its NumPy equivalent.
        # Time named operation
        start_time = time.time()
        for _ in range(iterations):
            named_result = named_op()
        named_time = (time.time() - start_time) / iterations
        
        # Time NumPy operation
        start_time = time.time()
        for _ in range(iterations):
            numpy_result = numpy_op()
        numpy_time = (time.time() - start_time) / iterations
        
        overhead_ratio = named_time / numpy_time if numpy_time > 0 else float('inf')
        return overhead_ratio

    def test_transpose_performance(self):
        # Benchmark transpose operations.
        def named_transpose():
            return self.named_large.transpose_named(['time', 'batch', 'features'])
        
        def numpy_transpose():
            return self.large_array.transpose((2, 0, 1))
        
        overhead = self.benchmark_operation("Transpose", named_transpose, numpy_transpose)
        
        # Overhead should be reasonable (less than 10x for this operation)
        assert_(overhead < 10.0, "Transpose overhead is too high")

    def test_reduction_performance(self):
        # Benchmark reduction operations.
        def named_sum():
            return self.named_large.sum_named('batch')
        
        def numpy_sum():
            return np.sum(self.large_array, axis=0)
        
        overhead = self.benchmark_operation("Sum", named_sum, numpy_sum)
        assert_(overhead < 5.0, "Sum overhead is too high")

    def test_memory_overhead(self):
        # Test memory overhead of NamedArray wrapper.
        import sys
        
        # Measure memory usage
        numpy_size = sys.getsizeof(self.large_array)
        named_size = sys.getsizeof(self.named_large)
        wrapper_overhead = named_size - numpy_size
        
        # Overhead should be minimal compared to array size
        overhead_ratio = wrapper_overhead / numpy_size
        assert_(overhead_ratio < 0.01, "Memory overhead is too high")


# Parametrized tests for different dtypes
@pytest.mark.parametrize('dtype', num_dts, ids=num_ids)
class TestNamedArrayDtypes:
    # Test NamedArray with different numeric dtypes.

    def test_dtype_preservation(self, dtype):
        # Test that dtypes are preserved correctly.
        arr = np.array([1, 2, 3, 4], dtype=dtype)
        named_arr = NamedArray(arr, ['time'])
        assert_equal(named_arr.dtype, dtype)
        assert_equal(named_arr._array.dtype, dtype)

    def test_dtype_operations(self, dtype):
        # Test operations preserve dtypes.
        arr = np.array([[1, 2], [3, 4]], dtype=dtype)
        named_arr = NamedArray(arr, ['batch', 'features'])
        
        # Test transpose
        transposed = named_arr.transpose_named(['features', 'batch'])
        assert_equal(transposed.dtype, dtype)
        
        # Test reduction
        if dtype.kind in 'biufc':  # numeric types
            summed = named_arr.sum_named('batch')
            # Note: sum might promote integer types
            assert_(summed.dtype.kind in 'biufc')


def test_real_world_scenarios():
    # Test real-world usage scenarios.
    
    # Machine Learning batch processing
    batch_data = random_named({
        'batch': 32,
        'height': 224,
        'width': 224,
        'channels': 3
    })
    assert_equal(batch_data.shape, (32, 224, 224, 3))
    assert_equal(batch_data.dim_names, ['batch', 'height', 'width', 'channels'])
    
    # Transpose for different framework conventions
    hwc_to_chw = batch_data.transpose_named(['batch', 'channels', 'height', 'width'])
    assert_equal(hwc_to_chw.shape, (32, 3, 224, 224))
    
    # Time series data
    time_series = random_named({'time': 1000, 'sensors': 10})
    daily_avg = time_series.mean_named('time')
    assert_equal(daily_avg.shape, (10,))
    assert_equal(daily_avg.dim_names, ['sensors'])