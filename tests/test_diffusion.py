import unittest
import _helpers
import torch
from impressionist.models.conditional_unet import ConditionalUNet
from impressionist.models.diffusion import GaussianDiffusion


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        self.diffusion = GaussianDiffusion(ConditionalUNet(8, [1, 2], 4), timesteps=8)

    def test_schedule_and_q_sample_formula(self):
        d = self.diffusion
        self.assertTrue(torch.allclose(d.alpha_bars, torch.cumprod(d.alphas, 0)))
        self.assertTrue(torch.all(d.alpha_bars[:-1] > d.alpha_bars[1:]))
        x0, noise = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
        t = torch.tensor([0, 7])
        expected = d.alpha_bars[t].sqrt()[:, None, None, None] * x0 + (1-d.alpha_bars[t]).sqrt()[:, None, None, None] * noise
        actual = d.q_sample(x0, t, noise)
        self.assertEqual(actual.shape, x0.shape)
        torch.testing.assert_close(actual, expected)

    def test_q_sample_zero_noise_and_generated_noise(self):
        x0 = torch.ones(1, 3, 16, 16)
        t = torch.tensor([3])
        expected = x0 * self.diffusion.sqrt_alpha_bars[3]
        torch.testing.assert_close(self.diffusion.q_sample(x0, t, torch.zeros_like(x0)), expected)
        self.assertEqual(self.diffusion.q_sample(x0, t).shape, x0.shape)

    def test_invalid_timestep_and_noise_shape(self):
        x0 = torch.zeros(1, 3, 16, 16)
        for timestep in (-1, 8):
            with self.assertRaises(ValueError):
                self.diffusion.q_sample(x0, torch.tensor([timestep]))
        with self.assertRaises(ValueError):
            self.diffusion.q_sample(x0, torch.tensor([0]), torch.zeros(3, 16, 16))

    def test_sampling_seed_bounds_and_mode_restore(self):
        condition = torch.zeros(1, 3, 16, 16)
        self.diffusion.train()
        rng = torch.get_rng_state().clone()
        first = self.diffusion.sample(condition, 9, progress=False)
        second = self.diffusion.sample(condition, 9, progress=False)
        other = self.diffusion.sample(condition, 10, progress=False)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(torch.equal(first, other))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(self.diffusion.training)
        self.assertEqual(first.shape, condition.shape)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(first.min() >= -1 and first.max() <= 1)

    def test_final_reverse_step_has_zero_variance(self):
        self.assertEqual(self.diffusion.posterior_variance[0].item(), 0)
        torch.testing.assert_close(self.diffusion.posterior_mean_x0[0], torch.tensor(1.0))
        torch.testing.assert_close(self.diffusion.posterior_mean_xt[0], torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
