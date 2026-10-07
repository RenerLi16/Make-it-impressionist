import unittest
import _helpers
import torch
from impressionist.evaluation.metrics import image_metrics


class MetricsTests(unittest.TestCase):
    def test_identity(self):
        x = torch.rand(2, 3, 16, 16)
        metrics = image_metrics(x, x)
        torch.testing.assert_close(metrics["l1"], torch.zeros(2))
        torch.testing.assert_close(metrics["mse"], torch.zeros(2))
        torch.testing.assert_close(metrics["ssim"], torch.ones(2))
        self.assertTrue(torch.isposinf(metrics["psnr"]).all())

    def test_known_errors(self):
        x, y = torch.zeros(1, 3, 16, 16), torch.full((1, 3, 16, 16), 0.5)
        metrics = image_metrics(x, y)
        self.assertAlmostEqual(metrics["l1"].item(), 0.5)
        self.assertAlmostEqual(metrics["mse"].item(), 0.25)
        self.assertAlmostEqual(metrics["psnr"].item(), 6.0206, places=4)
        self.assertLess(metrics["ssim"].item(), 0.01)


if __name__ == "__main__":
    unittest.main()
