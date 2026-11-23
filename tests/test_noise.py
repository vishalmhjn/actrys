import numpy as np

from actrys.noise import add_noise, clip_negative
from reference import add_noise_loop


def test_add_noise_matches_reference_loop():
    x = np.array([0, 3, 7.5, 120, 999, 42.2])
    np.random.seed(7)
    expected = add_noise_loop(x, 0.3, mu=1.2)
    np.random.seed(7)
    got = add_noise(x, 0.3, mu=1.2, rng=np.random.mtrand._rand)
    assert got.tolist() == expected


def test_add_noise_without_noise_truncates():
    out = add_noise(np.array([1.9, 2.5, 3.0]), 0, mu=1, rng=np.random.default_rng(0))
    assert out.tolist() == [1, 2, 3]


def test_add_noise_column_vector():
    out = add_noise(np.arange(6).reshape(-1, 1), 0.1, rng=np.random.default_rng(0))
    assert out.shape == (6,)


def test_clip_negative():
    assert clip_negative(np.array([-1, 0, 2])).tolist() == [0, 0, 2]
