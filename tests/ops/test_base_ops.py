from clowder.ops import base_ops

class TestDimension:
    def test_dimension(self):
        dim = base_ops.Dimension(12)
        assert 12 == dim.value
        assert 12 == int(dim)
        assert dim == base_ops.Dimension(12)
        assert base_ops.Dimension(15) == dim + base_ops.Dimension(3)
        assert base_ops.Dimension(15) == base_ops.Dimension(3) + dim
        assert base_ops.Dimension(9) == dim - 3
        assert base_ops.Dimension(1) == 13 - dim