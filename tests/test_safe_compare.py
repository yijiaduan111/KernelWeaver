import unittest
from unittest.mock import patch

import torch
from kernelbench.safe_compare import allclose, difference_stats, tensor_chunks


class SafeCompareTests(unittest.TestCase):
    def test_matches_torch_for_equal_mismatch_tolerances_and_nonfinite(self):
        for dtype in (torch.float32, torch.float64):
            for transposed in (False, True):
                for equal_nan in (False, True):
                    for error in (0, 0.00001, 2, float("nan"), float("inf")):
                        a = torch.linspace(-2, 2, 35, dtype=dtype).reshape(5, 7)
                        b = a.clone()
                        b[-1, -1] += error
                        if transposed:
                            a, b = a.t(), b.t()
                        expected = torch.allclose(a, b, rtol=.01, atol=.01, equal_nan=equal_nan)
                        self.assertEqual(allclose(a, b, rtol=.01, atol=.01, equal_nan=equal_nan, chunk_elements=3), expected)

    def test_views_are_bounded_without_contiguous_copy(self):
        a = torch.arange(210).reshape(5, 6, 7).permute(2, 0, 1)
        chunks = list(tensor_chunks(a, a, 4))
        self.assertEqual(sum(x.numel() for x, _ in chunks), a.numel())
        self.assertTrue(all(x.numel() <= 4 for x, _ in chunks))
        self.assertTrue(all(x.untyped_storage().data_ptr() == a.untyped_storage().data_ptr() for x, _ in chunks))

    def test_empty_scalar_and_shape_broadcast_keep_torch_behavior(self):
        for a, b in [(torch.empty(0), torch.empty(0)), (torch.tensor(1.), torch.tensor(1.)),
                     (torch.ones(3, 2), torch.ones(1, 2))]:
            self.assertEqual(allclose(a, b, chunk_elements=1), torch.allclose(a, b))

    def test_nan_inf_semantics(self):
        a = torch.tensor([1., float('inf'), -float('inf'), float('nan')])
        self.assertFalse(allclose(a, a, chunk_elements=1))
        self.assertTrue(allclose(a, a, equal_nan=True, chunk_elements=1))

    def test_diagnostics_keep_max_and_mean(self):
        a, b = torch.arange(100, dtype=torch.float32), torch.zeros(100)
        with patch.dict('os.environ', {'KERNELBENCH_CHECK_CHUNK_ELEMENTS': '7'}):
            maximum, mean = difference_stats(a, b)
        self.assertEqual(maximum, 99.)
        self.assertEqual(mean, 49.5)
