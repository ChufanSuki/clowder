import collections

LossOutput = collections.namedtuple("loss_output", ["loss", "extra"])

class Dimension(object):
    def __init__(self, value) -> None:
        if isinstance(value, int):
            if value < 0:
                raise ValueError("Dimension %d must be >= 0" % value)
            self._value = value
        elif value is None:
            self._value = None
        elif isinstance(value, Dimension):
            self._value = value._value
        else:
            try:
                self._value = int(value.__index__())
            except AttributeError:
                raise TypeError(
                    "Dimension value must be integer or None or have "
                    "an __index__ method, got value '{0!r}' with type '{1!r}'".format(
                        value, type(value))) from None
        if self._value < 0:
            raise ValueError("Dimension %d must be >= 0" % self._value)
        
    def merge_with(self, other):
        other = as_dimension(other)
        self.assert_is_compatible_with(other)
        if self._value is None:
            return Dimension(other.value)
        else:
            return Dimension(self._value)
    
    def assert_is_compatible_with(self, other):
        other = as_dimension(other)
        return (self._value is None or other.value is None or
            self._value == other.value)
    
    def __int__(self):
        return self._value
    
    def __eq__(self, other):
        """Returns true if `other` has the same known value as this Dimension."""
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return None
        return self._value == other.value
    
    def __radd__(self, other):
        """Returns the sum of `other` and `self`.

        Args:
        other: Another Dimension, or a value accepted by `as_dimension`.

        Returns:
        A Dimension whose value is the sum of `self` and `other`.
        """
        return self + other
    
    def __add__(self, other):
        """Returns the sum of `self` and `other`.

        Dimensions are summed as follows:

        ```python
        Dimension(m)    + Dimension(n)     ==
        Dimension(m + n)
        Dimension(m)    + Dimension(None)  # equiv. to
        Dimension(None)
        Dimension(None) + Dimension(n)     # equiv. to
        Dimension(None)
        Dimension(None) + Dimension(None)  # equiv. to
        Dimension(None)
        ```

        Args:
        other: Another Dimension, or a value accepted by `as_dimension`.

        Returns:
        A Dimension whose value is the sum of `self` and `other`.
        """
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value + other.value)
    
    def __sub__(self, other):
        """Returns the subtraction of `other` from `self`.

        Dimensions are subtracted as follows:

        ```python
        Dimension(m)    - Dimension(n)     ==
        Dimension(m - n)
        Dimension(m)    - Dimension(None)  # equiv. to
        Dimension(None)
        Dimension(None) - Dimension(n)     # equiv. to
        Dimension(None)
        Dimension(None) - Dimension(None)  # equiv. to
        Dimension(None)
        ```

        Args:
        other: Another Dimension, or a value accepted by `as_dimension`.

        Returns:
        A Dimension whose value is the subtraction of `other` from `self`.
        """
        try:
            other = as_dimension(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(self._value - other.value)
    
    def __rsub__(self, other):
        """Returns the subtraction of `self` from `other`.

        Args:
        other: Another Dimension, or a value accepted by `as_dimension`.

        Returns:
        A Dimension whose value is the subtraction of `self` from `other`.
        """
        other = as_dimension(other)
        if self._value is None or other.value is None:
            return Dimension(None)
        else:
            return Dimension(other.value - self._value)
    
    @property
    def value(self):
        """The value of this dimension, or None if it is unknown."""
        return self._value
    
    def is_compatible_with(self, other):
        """Returns true if `other` is compatible with this Dimension.

        Two known Dimensions are compatible if they have the same value.
        An unknown Dimension is compatible with all other Dimensions.

        Args:
        other: Another Dimension.

        Returns:
        True if this Dimension and `other` are compatible.
        """
        other = as_dimension(other)
        return (self._value is None or other.value is None or
                self._value == other.value)
        

def as_dimension(value):
    """Converts the given value to a Dimension.

    A Dimension input will be returned unmodified.
    An input of `None` will be converted to an unknown Dimension.
    An integer input will be converted to a Dimension with that value.

    Args:
        value: The value to be converted.

    Returns:
        A Dimension corresponding to the given value.
    """
    if isinstance(value, Dimension):
        return value
    else:
        return Dimension(value)

def as_shape(shape):
    """Converts the given object to a Shape."""
    if isinstance(shape, Shape):
        return shape
    else:
        return Shape(shape)


class Shape:
    def __init__(self, dims) -> None:
        """Creates a new TensorShape with the given dimensions.

        Args:
            dims: A list of Dimensions, or None if the shape is unspecified.

        Raises:
            TypeError: If dims cannot be converted to a list of dimensions.
        """
        if isinstance(dims, (tuple, list)):
            self._dims = tuple(as_dimension(d) for d in dims)
        elif dims is None:
            self._dims = None
        elif isinstance(dims, Shape):
            self._dims = dims._dims
    
    @property
    def rank(self):
        """Returns the rank of this shape, or None if it is unspecified."""
        if self._dims is not None:
            return len(self._dims)
        return None
    
    def assert_has_rank(self, rank):
        """Raises an exception if `self` is not compatible with the given `rank`.

        Args:
        rank: An integer.

        Raises:
        ValueError: If `self` does not represent a shape with the given `rank`.
        """
        if self.rank not in (None, rank):
            raise ValueError("Shape %s must have rank %d" % (self, rank))
    
    def assert_same_rank(self, other):
        """Raises an exception if `self` and `other` do not have compatible ranks.

        Args:
            other: Another `TensorShape`.

        Raises:
            ValueError: If `self` and `other` do not represent shapes with the
            same rank.
        """
        other = as_shape(other)
        if self.rank is not None and other.rank is not None:
            if self.rank != other.rank:
                raise ValueError(f"Shapes {self} and {other} must have the same rank")
    
    def as_list(self):
        """Returns a list of integers or `None` for each dimension.

        Returns:
        A list of integers or `None` for each dimension.

        Raises:
        ValueError: If `self` is an unknown shape with an unknown rank.
        """
        if self._dims is None:
            raise ValueError("as_list() is not defined on an unknown Shape.")
        return list(self._dims)
    
    def merge_with(self, other):
        """Returns a `TensorShape` combining the information in `self` and `other`.

        The dimensions in `self` and `other` are merged element-wise,
        according to the rules below:

        ```python
        Dimension(n).merge_with(Dimension(None)) == Dimension(n)
        Dimension(None).merge_with(Dimension(n)) == Dimension(n)
        Dimension(None).merge_with(Dimension(None)) == Dimension(None)
        # raises ValueError for n != m
        Dimension(n).merge_with(Dimension(m))
        ```
        >> ts = Shape([1,2])
        >> ot1 = Shape([1,2])
        >> ts.merge_with(ot).as_list()
        [1,2]

        >> ot2 = Shape([1,None])
        >> ts.merge_with(ot2).as_list()
        [1,2]

        >> ot3 = Shape([None, None])
        >> ot3.merge_with(ot2).as_list()
        [1, None]

        Args:
        other: Another `Shape`.

        Returns:
        A `Shape` containing the combined information of `self` and
        `other`.

        Raises:
        ValueError: If `self` and `other` are not compatible.
        """
        other = as_shape(other)
        if self.as_list() is None:
            return other
        if other.as_list() is None:
            return self
        else:
            try:
                self.assert_same_rank(other)
                new_dims = [dim.merge_with(other_dim) for dim, other_dim in zip(self.as_list(), other.as_list())]
                return Shape(new_dims)
            except ValueError:
                raise ValueError(f"shape {self} and {other} are not compatible")

def assert_rank_and_shape_compatibility(tensors, rank):
    """Asserts that the tensors have the correct rank and compatible shapes.

    Shapes (of equal rank) are compatible if corresponding dimensions are all
    equal or unspecified. E.g. `[2, 3]` is compatible with all of `[2, 3]`,
    `[None, 3]`, `[2, None]` and `[None, None]`.

    Args:
        tensors: List of tensors.
        rank: A scalar specifying the rank that the tensors passed need to have.

    Raises:
        ValueError: If the list of tensors is empty or fail the rank and mutual
        compatibility asserts.
    """
    if not tensors:
        raise ValueError("List of tensors should be non-empty.")
    
    union_of_shapes = Shape(None)
    for tensor in tensors:
        tensor_shape = Shape(tensor.shape)
        tensor_shape.assert_has_rank(rank)
        union_of_shapes = union_of_shapes.merge_with(tensor_shape)
    