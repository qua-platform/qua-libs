import math
import numpy as np
from typing import Any, Tuple


def get_max_shape(lst: Any) -> Tuple[int, ...]:
    """
    Recursively determine the maximum shape needed to pad all sublists
    so that the entire nested structure becomes a regular ndarray.
    """
    if not isinstance(lst, list):
        return ()
    # First dimension: length of this list
    first_dim = len(lst)
    # Compute shapes of all children
    child_shapes = [get_max_shape(item) for item in lst]
    if not child_shapes:
        return (first_dim,)
    # Make all child shapes the same length by padding with zeros
    max_depth = max(len(shape) for shape in child_shapes)
    padded = [shape + (0,) * (max_depth - len(shape)) for shape in child_shapes]
    # Compute the max size in each subsequent dimension
    child_dims = tuple(
        max(padded[i][d] for i in range(len(padded)))
        for d in range(max_depth)
    )
    return (first_dim,) + child_dims


def make_filled(shape: Tuple[int, ...], fill: Any) -> Any:
    """
    Build a nested list of given shape, filled with `fill`.
    """
    if not shape:
        return fill
    size, *rest = shape
    return [make_filled(tuple(rest), fill) for _ in range(size)]


def pad_to_ndarray(lst: Any, padding_value: Any = None) -> Any:
    """
    Recursively pad nested lists with `padding_value` so that at every depth
    all sublists have the same length. After padding, np.array(...) yields
    a regular ndarray.
    """
    shape = get_max_shape(lst)

    def _pad(x: Any, dims: Tuple[int, ...]) -> Any:
        # No further dims: return scalar or list as-is
        if not dims:
            return x
        expected_len, *rest = dims
        # Ensure x is a list
        if not isinstance(x, list):
            x = [x]
        # Pad this level
        if len(x) < expected_len:
            pad_block = make_filled(tuple(rest), padding_value)
            x = x + [pad_block] * (expected_len - len(x))
        # Recurse into each element
        return [_pad(item, tuple(rest)) for item in x]

    return _pad(lst, shape)


# ====== Tests ======
def test_pad_to_ndarray():
    print("\n=== pad_to_ndarray Tests ===\n")

    tests = []
    # 1) Simple 2D ragged
    tests.append(([[1, 2, 3], [4, 5]], (2, 3)))
    # 2) 3D ragged
    tests.append(([[[1], [2, 3]], [[4, 5, 6]]], (2, 2, 3)))
    # 3) Flat list
    tests.append(([1, 2, 3, 4], (4,)))
    # 4) Mixed scalars and lists at same level
    tests.append(([1, [2, 3], [4]], (3, 2)))
    # 5) Deeply nested ragged
    deep = [[[[1], []], [[2, 3]]], [[[4, 5, 6]]]]
    # Expected shape: (2, 2, 2, 3)
    tests.append((deep, (2, 2, 2, 3)))
    # 6) Empty root
    tests.append(([], (0,)))
    # 7) List of empty lists
    tests.append(([[], []], (2, 0)))
    # 8) Single scalar (0-D array)
    tests.append((5, ()))

    for idx, (inp, expected_shape) in enumerate(tests, 1):
        out = pad_to_ndarray(inp)
        arr = np.array(out)
        print(f"Test {idx}: input={inp}")
        print(f"Result array shape: {arr.shape}, expected: {expected_shape}")
        assert arr.shape == expected_shape, \
            f"Shape mismatch: got {arr.shape}, want {expected_shape}"
        print(f"✓ Test {idx} passed\n")

    print("=== All tests passed! ===\n")

def split_list_by_integer_count(list_of_lists, max_integers):
    """
    Split a list of lists of integers into chunks where each chunk has
    roughly the same total number of integers, with a maximum total per chunk.
    
    Args:
        list_of_lists: List of lists, where each inner list contains integers
        max_integers: Maximum total number of integers allowed per chunk
    
    Returns:
        List of chunks, where each chunk is a list of the original inner lists,
        and each chunk has at most max_integers total integers
    """
    if max_integers <= 0:
        raise ValueError("max_integers must be positive")
    
    if not list_of_lists:
        return []
    
    # Calculate the size (number of integers) for each inner list
    list_sizes = [len(inner_list) for inner_list in list_of_lists]
    
    if max_integers < max(list_sizes):
        raise ValueError("max_integers must be greater than the maximum size of any inner list")
    
    total_integers = sum(list_sizes)
    
    # Calculate minimum number of chunks needed
    min_chunks = math.ceil(total_integers / max_integers)
    
    # Target number of integers per chunk (as even as possible)
    target_per_chunk = total_integers / min_chunks
    
    chunks = []
    current_chunk = []
    current_count = 0
    
    for i, inner_list in enumerate(list_of_lists):
        inner_size = list_sizes[i]
        
        # Check if adding this list would exceed max_integers
        if current_count + inner_size > max_integers:
            # Start a new chunk if current chunk is not empty
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_count = 0
        
        # Add the current inner list to the current chunk
        current_chunk.append(inner_list)
        current_count += inner_size
        
        # Check if we should start a new chunk to balance sizes
        # (only if we haven't exceeded max_integers and we're above target)
        remaining_chunks = min_chunks - len(chunks)
        if (remaining_chunks > 1 and 
            current_count >= target_per_chunk and 
            i < len(list_of_lists) - 1):  # Don't start new chunk on last item
            chunks.append(current_chunk)
            current_chunk = []
            current_count = 0
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

if __name__ == "__main__":
    test_pad_to_ndarray()