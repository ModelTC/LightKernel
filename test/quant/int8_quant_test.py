import unittest
import torch
from vllm import _custom_ops as ops
from lightllm_kernel.ops import per_token_quant_int8
from test.utils import benchmark, error


class TestQuantINT8(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.tokens = [1024, 13325]
        self.hiddenDims = [3, 256, 257, 511, 1023, 1024, 1025, 1032, 3200, 3201, 3208, 12800]
        self.device = "cuda"
        self.dtypes = [torch.bfloat16, torch.float16]

    def test_accuracy(self):
        """Test the accuracy of per_token_quant"""
        for dtype in self.dtypes:
            for token in self.tokens:
                for hiddenDim in self.hiddenDims:
                    with self.subTest(shape=[token, hiddenDim], dtype=dtype):
                        input = torch.rand(size=[token, hiddenDim], device=self.device, dtype=dtype) - 0.5
                        y_real, scales_real, _ = ops.scaled_int8_quant(
                            input.contiguous()
                        )
                        y_pred, scales_pred = per_token_quant_int8(input)
                        self.assertTrue(
                            error(scales_real, scales_pred) < 0.01,
                            f"Accuracy test failed for size {token}, {hiddenDim}."
                            f"scales_real={scales_real}, scales_pred={scales_pred}",
                        )
                        self.assertTrue(
                            error(y_real, y_pred) < 0.01,
                            f"Accuracy test failed for size {token}, {hiddenDim}. y_real={y_real}, y_pred={y_pred}",
                        )

    def test_performance(self):
        """Test the performance of per_token_quant"""
        for dtype in self.dtypes:
            for token in self.tokens:
                for size in self.hiddenDims:
                    with self.subTest(shape=[token, size], dtype=dtype):
                        input = torch.rand(size=[token, size], device=self.device, dtype=dtype) - 0.5
                        shape = [[token, size]]
                        tflops = token * size / 1024 ** 4
                        benchmark(per_token_quant_int8, shape, tflops, 100, input)
                        benchmark(ops.scaled_int8_quant, shape, tflops, 100, input)


if __name__ == "__main__":
    unittest.main()
