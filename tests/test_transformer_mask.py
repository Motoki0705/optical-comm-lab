import numpy as np
import pytest

from experiments.transformer_nlc.mask import additive_mask, block_mask, individual_mask


def reference_mask(t, b, rho):
    # Coordinate-by-coordinate definition, independent of slice union code.
    ell = 2 * t - 8
    q = ell // 2
    n = b + ell
    return np.array([[any(i <= row <= i + ell and i <= col <= i + ell
                          and abs((row - i - q) * (col - i - q)) <= rho * q
                          for i in range(b))
                      for col in range(n)] for row in range(n)], dtype=bool)


@pytest.mark.parametrize("t,b,rho", [(4, 1, 2.6), (8, 1, 0), (8, 5, 0.4), (10, 3, 2.6)])
def test_product_union_matches_independent_coordinate_reference(t, b, rho):
    actual = block_mask(t, b, rho)
    np.testing.assert_array_equal(actual, reference_mask(t, b, rho))
    np.testing.assert_array_equal(actual, actual.T)
    np.testing.assert_array_equal(actual, actual[::-1, ::-1])
    assert actual.any(axis=-1).all()
    assert actual[0].any() and actual[-1].any()


def test_individual_center_row_column_and_inclusive_edges_are_preserved():
    t, rho = 32, 2.6
    mask = individual_mask(t, rho)
    q = (2 * t - 8) // 2
    assert mask.shape == (2 * q + 1, 2 * q + 1)
    assert mask[q].all() and mask[:, q].all()
    assert mask[0, q] and mask[-1, q]
    assert not mask[0, 0] and not mask[-1, -1]
    np.testing.assert_array_equal(block_mask(t, 1, rho), mask)
    # Exact threshold is inclusive, not a strict hyperbola.
    exact = individual_mask(8, 1.0)
    assert exact[4 + 2, 4 + 2]
    assert not exact[4 + 2, 4 + 3]


@pytest.mark.parametrize("t,b,rho,density", [
    (32, 1, 2.6, 0.308095),
    (64, 128, 2.6, 0.342709),
    (64, 128, 0.4, 0.314061),
    (64, 1, 2.6, 0.178198),
])
def test_fixed_spec_densities_including_unresolved_figure4b(t, b, rho, density):
    measured = block_mask(t, b, rho).mean()
    assert measured == pytest.approx(density, abs=0.5e-6)
    if t == 64 and b == 1:
        assert measured != pytest.approx(0.19, abs=0.005)


def test_additive_mask_has_only_zero_or_negative_infinity_and_finite_softmax():
    allowed = block_mask(12, 1, 2.6)
    additive = additive_mask(allowed, dtype=np.float64)
    assert np.all(additive[allowed] == 0)
    assert np.isneginf(additive[~allowed]).all()
    logits = np.arange(allowed.size, dtype=float).reshape(allowed.shape) / allowed.size
    logits += additive
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1)
    assert (probabilities[~allowed] == 0).all()
    assert block_mask(8, 3, None).all()


@pytest.mark.parametrize("kwargs", [{"t": 0}, {"t": 3}, {"b": 0}, {"t": True},
                                    {"k": 8}, {"k": 7}, {"rho": -1},
                                    {"rho": float("nan")}, {"rho": float("inf")},
                                    {"rho": True}])
def test_invalid_geometry_or_rho_rejected(kwargs):
    options = {"t": 8, "b": 1, "rho": 2.6, **kwargs}
    with pytest.raises(ValueError):
        block_mask(**options)


def test_additive_conversion_rejects_wrong_type_shape_empty_rows_and_integer_dtype():
    for invalid in (np.ones((2, 3), dtype=bool), np.ones((2, 2)),
                    np.array([[True, False], [False, False]]), np.empty((0, 0), dtype=bool)):
        with pytest.raises(ValueError):
            additive_mask(invalid)
    with pytest.raises(ValueError):
        additive_mask(np.ones((1, 1), dtype=bool), dtype=np.int32)
