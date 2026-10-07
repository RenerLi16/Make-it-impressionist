import unittest
import _helpers
import torch
from impressionist.models.conditional_unet import ConditionalUNet
from impressionist.models.embeddings import SinusoidalTimeEmbedding


class ModelTests(unittest.TestCase):
    def test_batch_one_and_multiple_sizes(self):
        model = ConditionalUNet(8, [1, 2, 4], 4)
        for batch, size in ((1, 16), (2, 32), (1, 128)):
            with self.subTest(batch=batch, size=size), torch.no_grad():
                x = torch.randn(batch, 3, size, size)
                self.assertEqual(model(x, torch.zeros(batch, dtype=torch.long), x).shape, x.shape)

    def test_batch_one_smallest_spatial_map(self):
        model = ConditionalUNet(4, [1, 2], 2)
        x = torch.randn(1, 3, 2, 2)
        self.assertEqual(model(x, torch.tensor([0]), x).shape, x.shape)

    def test_condition_encoder_receives_gradients(self):
        model = ConditionalUNet(8, [1, 2], 4)
        x = torch.randn(1, 3, 16, 16)
        model(x, torch.tensor([3]), x).square().mean().backward()
        grads = [p.grad for p in model.condition_encoder.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_condition_changes_prediction_and_cache_matches(self):
        model = ConditionalUNet(8, [1, 2], 4).eval()
        x, condition, t = torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16), torch.tensor([1])
        with torch.no_grad():
            normal = model(x, t, condition)
            cached = model(x, t, condition_features=model.encode_condition(condition))
            changed = model(x, t, torch.zeros_like(condition))
        torch.testing.assert_close(normal, cached)
        self.assertFalse(torch.allclose(normal, changed))

    def test_bad_shapes_and_odd_embedding_width(self):
        self.assertEqual(SinusoidalTimeEmbedding(7)(torch.tensor([1])).shape, (1, 7))
        model = ConditionalUNet(8, [1, 2, 4], 4)
        with self.assertRaisesRegex(ValueError, "multiples"):
            x = torch.zeros(1, 3, 15, 15)
            model(x, torch.tensor([1]), x)
        with self.assertRaisesRegex(ValueError, "int64"):
            x = torch.zeros(1, 3, 16, 16)
            model(x, torch.tensor([1.0]), x)


if __name__ == "__main__":
    unittest.main()
