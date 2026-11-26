import os
import tempfile

import numpy as np
from PIL import Image

from src.mask import create_square_mask, create_scratch_mask
from src.image_processing import load_grayscale, save_image, restore_pixels


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


def test_load_grayscale_rgb():
    p = os.path.join(tempfile.mkdtemp(), "rgb.png")
    Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(p)
    a = load_grayscale(p)
    assert a.dtype == np.float64 and a.shape == (8, 8)
    assert 0.0 <= a.min() and a.max() <= 255.0
    assert np.allclose(a, 200.0)   # RGB(200,200,200) -> L = 200


def test_save_image_clips_and_rounds():
    p = os.path.join(tempfile.mkdtemp(), "out.png")
    save_image(np.array([[-10.0, 127.6], [300.0, 255.4]]), p)
    b = np.asarray(Image.open(p))
    assert b.dtype == np.uint8
    assert b.flatten().tolist() == [0, 128, 255, 255]


def test_restore_pixels():
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    img_orig = img.copy()
    mask = create_square_mask((4, 4), 1, 1, 2)
    unknown = np.argwhere(mask == 1)
    x = np.array([99.0, 98.0, 97.0, 96.0])
    out = restore_pixels(img, mask, unknown, x)
    assert np.allclose(out[mask == 1], x)                 # đúng thứ tự argwhere
    assert np.allclose(out[mask == 0], img[mask == 0])    # giữ vùng ngoài
    assert np.allclose(img, img_orig)                     # không đổi ảnh gốc


def test_images_smooth_vs_edges():
    i1 = load_grayscale("data/original/image1_smooth_256.png")
    i2 = load_grayscale("data/original/image2_edges_256.png")
    for a in (i1, i2):
        assert a.dtype == np.float64 and a.shape == (256, 256)
    d1 = np.abs(np.diff(i1, axis=0)).max(), np.abs(np.diff(i1, axis=1)).max()
    assert max(d1) < 10                          # ảnh 1 gần như phẳng từng pixel
    e2 = (np.abs(np.diff(i2, axis=0)) > 50).sum() + (np.abs(np.diff(i2, axis=1)) > 50).sum()
    assert e2 >= 500                             # ảnh 2 nhiều rìa sắc nét


def test_tn1_masks_fixed():
    specs = [("mask_2x50.npy", 100, (slice(127, 129), slice(103, 153))),
             ("mask_10x10.npy", 100, (slice(123, 133), slice(123, 133))),
             ("mask_30x30.npy", 900, (slice(113, 143), slice(113, 143)))]
    for name, n, region in specs:
        m = np.load(f"data/masks/{name}")
        assert m.dtype == np.uint8 and m.shape == (256, 256)
        assert set(np.unique(m).tolist()) <= {0, 1} and m.sum() == n
        assert m[region].sum() == n               # đúng vị trí căn giữa, lệch 1 pixel cũng fail


def run_tests():
    test_square_mask(); test_scratch_mask(); test_edge_guard()
    test_load_grayscale_rgb(); test_save_image_clips_and_rounds(); test_restore_pixels()
    test_images_smooth_vs_edges(); test_tn1_masks_fixed()
    print("ALL IMAGE/MASK TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
