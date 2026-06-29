from src.flatten import flatten


def test_full_flatten():
    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]


def test_no_nesting():
    assert flatten([1, 2, 3]) == [1, 2, 3]


def test_depth_zero_disables_flatten():
    assert flatten([1, [2, [3]]], depth=0) == [1, [2, [3]]]


def test_depth_one_flattens_one_level():
    assert flatten([1, [2, [3]]], depth=1) == [1, 2, [3]]


def test_depth_two_flattens_two_levels():
    assert flatten([1, [2, [3, [4]]]], depth=2) == [1, 2, 3, [4]]
