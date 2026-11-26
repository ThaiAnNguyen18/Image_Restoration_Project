import numpy as np
from src.mask import create_square_mask, create_scratch_mask


def test_square_mask():
    m = create_square_mask((10, 12), row=2, col=3, size=4)
    assert m.dtype == np.uint8 and m.shape == (10, 12)
    assert set(np.unique(m).tolist()) <= {0, 1}
    assert m[2:6, 3:7].sum() == 16 and m.sum() == 16
    m2 = create_square_mask((4, 4), row=1, col=1, size=2)   # đúng ca test_2x2
    assert m2.sum() == 4 and m2[1:3, 1:3].sum() == 4


def test_scratch_mask():
    m = create_scratch_mask((10, 12), row=4, col=1, height=2, width=6)
    assert m.dtype == np.uint8 and m.shape == (10, 12)
    assert m.sum() == 12 and m[4:6, 1:7].sum() == 12


def test_edge_guard():
    bad = [
        (create_square_mask, ((10, 12), 0, 3, 4)),    # row = 0
        (create_square_mask, ((10, 12), 7, 3, 4)),    # 7+4 = 11 > 9
        (create_scratch_mask, ((10, 12), 4, 0, 2, 6)),   # col = 0
        (create_scratch_mask, ((10, 12), 4, 7, 2, 6)),   # 7+6 = 13 > 11
    ]
    for fn, args in bad:
        try:
            fn(*args)
            assert False, f"phải raise ValueError: {fn.__name__}{args}"
        except ValueError:
            pass


def run_tests():
    test_square_mask(); test_scratch_mask(); test_edge_guard()
    print("ALL MASK TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
