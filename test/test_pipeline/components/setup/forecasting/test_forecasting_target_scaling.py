import unittest

import torch

from autoPyTorch.pipeline.components.setup.forecasting_target_scaling.base_target_scaler import BaseTargetScaler


class TestTargetScalar(unittest.TestCase):
    def test_target_no_scalar(self):
        X = {'dataset_properties': {}}
        scalar = BaseTargetScaler(scaling_mode='none')
        scalar = scalar.fit(X)
        X = scalar.transform(X)
        self.assertIsInstance(X['target_scaler'], BaseTargetScaler)

        past_targets = torch.rand([5, 6, 7])
        future_targets = torch.rand(([5, 3, 7]))

        past_observed_values = torch.rand([5, 6, 7]) > 0.5

        transformed_past_target, transformed_future_targets, loc, scale = scalar(
            past_targets, past_observed_values=past_observed_values, future_targets=future_targets)
        self.assertTrue(torch.equal(past_targets, transformed_past_target))
        self.assertTrue(torch.equal(future_targets, transformed_future_targets))
        self.assertIsNone(loc)
        self.assertIsNone(scale)

        _, transformed_future_targets, _, _ = scalar(past_targets)
        self.assertIsNone(transformed_future_targets)

    def test_target_mean_abs_scalar(self):
        X = {'dataset_properties': {}}
        scalar = BaseTargetScaler(scaling_mode='mean_abs')
        scalar = scalar.fit(X)
        X = scalar.transform(X)
        self.assertIsInstance(X['target_scaler'], BaseTargetScaler)

        past_targets = torch.vstack(
            [
                torch.zeros(10),
                torch.Tensor([0.] * 2 + [1.] * 5 + [2.] * 3),
                torch.ones(10) * 4
            ]
        ).unsqueeze(-1)
        past_observed_values = torch.vstack(
            [
                torch.Tensor([False] * 3 + [True] * 7),
                torch.Tensor([False] * 2 + [True] * 8),
                torch.Tensor([True] * 10)

            ]).unsqueeze(-1).bool()
        future_targets = torch.ones([3, 10, 1]) * 10

        transformed_past_target, transformed_future_targets, loc, scale = scalar(
            past_targets, past_observed_values=past_observed_values, future_targets=future_targets
        )

        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.allclose(transformed_past_target[1],
                                       torch.Tensor([0.] * 2 + [8. / 11.] * 5 + [16. / 11.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.ones([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 80. / 11.))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 2.5))

        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 11. / 8., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )
        self.assertIsNone(loc)

        transformed_past_target, transformed_future_targets, loc, scale = scalar(past_targets,
                                                                                 future_targets=future_targets)

        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.allclose(transformed_past_target[1],
                                       torch.Tensor([0.] * 2 + [10. / 11.] * 5 + [20. / 11.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.ones([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 100. / 11))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 2.5))

        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 1.1, 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )

        transformed_past_target_full, transformed_future_targets_full, loc_full, scale_full = scalar(
            past_targets, past_observed_values=torch.ones([2, 10, 1], dtype=torch.bool), future_targets=future_targets
        )

        self.assertTrue(torch.equal(transformed_past_target, transformed_past_target_full))
        self.assertTrue(torch.equal(transformed_future_targets_full, transformed_future_targets_full))
        self.assertTrue(torch.equal(scale, scale_full))

        self.assertIsNone(loc_full)

        _, _, _, scale = scalar(
            torch.Tensor([[1e-10, 1e-10, 1e-10], [1e-15, 1e-15, 1e-15]]).reshape([2, 3, 1])
        )
        self.assertTrue(torch.equal(scale.flatten(), torch.Tensor([1e-10, 1.])))

    def test_target_standard_scalar(self):
        X = {'dataset_properties': {}}
        scalar = BaseTargetScaler(scaling_mode='standard')
        scalar = scalar.fit(X)
        X = scalar.transform(X)
        self.assertIsInstance(X['target_scaler'], BaseTargetScaler)

        past_targets = torch.vstack(
            [
                torch.zeros(10),
                torch.Tensor([0.] * 2 + [1.] * 5 + [2.] * 3),
                torch.ones(10) * 4
            ]
        ).unsqueeze(-1)
        past_observed_values = torch.vstack(
            [
                torch.Tensor([False] * 3 + [True] * 7),
                torch.Tensor([False] * 2 + [True] * 8),
                torch.Tensor([True] * 10)

            ]).unsqueeze(-1).bool()
        future_targets = torch.ones([3, 10, 1]) * 10

        transformed_past_target, transformed_future_targets, loc, scale = scalar(
            past_targets, past_observed_values=past_observed_values, future_targets=future_targets
        )

        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.allclose(transformed_past_target[1],
                                       torch.Tensor([0.] * 2 + [-0.7246] * 5 + [1.2076] * 3).unsqueeze(-1), atol=1e-4))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.zeros([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.allclose(transformed_future_targets[1], torch.ones([10, 1]) * 16.6651))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 6.0000))

        self.assertTrue(
            torch.allclose(loc,
                           torch.Tensor([0., 11. / 8., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )

        self.assertTrue(
            torch.allclose(scale,
                           torch.Tensor([1., 0.5175, 1.]).reshape([len(past_targets), 1, past_targets.shape[-1]]),
                           atol=1e-4)
        )

        transformed_past_target, transformed_future_targets, loc, scale = scalar(past_targets,
                                                                                 future_targets=future_targets)

        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))

        self.assertTrue(torch.allclose(transformed_past_target[1],
                                       torch.Tensor([-1.4908] * 2 + [-0.1355] * 5 + [1.2197] * 3).unsqueeze(-1),
                                       atol=1e-4)
                        )
        self.assertTrue(torch.equal(transformed_past_target[2], torch.zeros([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.allclose(transformed_future_targets[1], torch.ones([10, 1]) * 12.0618, atol=1e-4))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 6.))

        self.assertTrue(
            torch.allclose(loc,
                           torch.Tensor([0., 1.1, 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]),
                           atol=1e-4
                           )
        )
        self.assertTrue(
            torch.allclose(scale,
                           torch.Tensor([1., 0.7379, 1.]).reshape([len(past_targets), 1, past_targets.shape[-1]]),
                           atol=1e-4
                           )
        )

        transformed_past_target_full, transformed_future_targets_full, loc_full, scale_full = scalar(
            past_targets, past_observed_values=torch.ones([2, 10, 1], dtype=torch.bool), future_targets=future_targets
        )
        self.assertTrue(torch.equal(transformed_past_target, transformed_past_target_full))
        self.assertTrue(torch.equal(transformed_future_targets_full, transformed_future_targets_full))
        self.assertTrue(torch.equal(loc, loc_full))
        self.assertTrue(torch.equal(scale, scale_full))

        _, _, _, scale = scalar(
            torch.Tensor([[1e-10, -1e-10, 1e-10], [1e-15, -1e-15, 1e-15]]).reshape([2, 3, 1])
        )
        self.assertTrue(torch.all(torch.isclose(scale.flatten(), torch.Tensor([1.1547e-10, 1.]))))

    def test_target_min_max_scalar(self):
        X = {'dataset_properties': {}}
        scalar = BaseTargetScaler(scaling_mode='min_max')
        scalar = scalar.fit(X)
        X = scalar.transform(X)
        self.assertIsInstance(X['target_scaler'], BaseTargetScaler)

        past_targets = torch.vstack(
            [
                torch.zeros(10),
                torch.Tensor([0.] * 2 + [1.] * 5 + [2.] * 3),
                torch.ones(10) * 4
            ]
        ).unsqueeze(-1)
        past_observed_values = torch.vstack(
            [
                torch.Tensor([False] * 3 + [True] * 7),
                torch.Tensor([False] * 2 + [True] * 8),
                torch.Tensor([True] * 10)

            ]).unsqueeze(-1).bool()
        future_targets = torch.ones([3, 10, 1]) * 10

        transformed_past_target, transformed_future_targets, loc, scale = scalar(
            past_targets, past_observed_values=past_observed_values, future_targets=future_targets
        )

        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.allclose(transformed_past_target[1],
                                       torch.Tensor([0.] * 2 + [0.] * 5 + [1.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.zeros([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 9))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 1.5))

        self.assertTrue(
            torch.equal(loc, torch.Tensor([0., 1., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )
        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 1., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )

        transformed_past_target, transformed_future_targets, loc, scale = scalar(past_targets,
                                                                                 future_targets=future_targets)
        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.equal(transformed_past_target[1],
                                    torch.Tensor([0.] * 2 + [0.5] * 5 + [1.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.zeros([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 5))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 1.5))
        self.assertTrue(
            torch.equal(loc, torch.Tensor([0., 0., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )
        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 2., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )

        transformed_past_target_full, transformed_future_targets_full, loc_full, scale_full = scalar(
            past_targets, past_observed_values=torch.ones([2, 10, 1], dtype=torch.bool), future_targets=future_targets
        )
        self.assertTrue(torch.equal(transformed_past_target, transformed_past_target_full))
        self.assertTrue(torch.equal(transformed_future_targets_full, transformed_future_targets_full))
        self.assertTrue(torch.equal(scale, scale_full))

        _, _, _, scale = scalar(
            torch.Tensor([[1e-10, 1e-10, 1e-10], [1e-15, 1e-15, 1e-15]]).reshape([2, 3, 1])
        )
        self.assertTrue(torch.equal(scale.flatten(), torch.Tensor([1e-10, 1.])))

    def test_target_max_abs_scalar(self):
        X = {'dataset_properties': {}}
        scalar = BaseTargetScaler(scaling_mode='max_abs')
        scalar = scalar.fit(X)
        X = scalar.transform(X)
        self.assertIsInstance(X['target_scaler'], BaseTargetScaler)

        past_targets = torch.vstack(
            [
                torch.zeros(10),
                torch.Tensor([0.] * 2 + [1.] * 5 + [2.] * 3),
                torch.ones(10) * 4
            ]
        ).unsqueeze(-1)
        past_observed_values = torch.vstack(
            [
                torch.Tensor([False] * 3 + [True] * 7),
                torch.Tensor([False] * 2 + [True] * 8),
                torch.Tensor([True] * 10)

            ]).unsqueeze(-1).bool()
        future_targets = torch.ones([3, 10, 1]) * 10

        transformed_past_target, transformed_future_targets, loc, scale = scalar(
            past_targets, past_observed_values=past_observed_values, future_targets=future_targets
        )

        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.allclose(transformed_past_target[1],
                                       torch.Tensor([0.] * 2 + [0.5] * 5 + [1.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.ones([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 5.))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 2.5))

        self.assertIsNone(loc)
        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 2., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )

        transformed_past_target, transformed_future_targets, loc, scale = scalar(past_targets,
                                                                                 future_targets=future_targets)
        self.assertTrue(torch.equal(transformed_past_target[0], torch.zeros([10, 1])))
        self.assertTrue(torch.equal(transformed_past_target[1],
                                    torch.Tensor([0.] * 2 + [0.5] * 5 + [1.] * 3).unsqueeze(-1)))
        self.assertTrue(torch.equal(transformed_past_target[2], torch.ones([10, 1])))

        self.assertTrue(torch.equal(transformed_future_targets[0], torch.ones([10, 1]) * 10))
        self.assertTrue(torch.equal(transformed_future_targets[1], torch.ones([10, 1]) * 5))
        self.assertTrue(torch.equal(transformed_future_targets[2], torch.ones([10, 1]) * 2.5))

        self.assertIsNone(loc)
        self.assertTrue(
            torch.equal(scale, torch.Tensor([1., 2., 4.]).reshape([len(past_targets), 1, past_targets.shape[-1]]))
        )

        transformed_past_target_full, transformed_future_targets_full, loc_full, scale_full = scalar(
            past_targets, past_observed_values=torch.ones([2, 10, 1], dtype=torch.bool), future_targets=future_targets
        )
        self.assertTrue(torch.equal(transformed_past_target, transformed_past_target_full))
        self.assertTrue(torch.equal(transformed_future_targets_full, transformed_future_targets_full))
        self.assertIsNone(loc_full)
        self.assertTrue(torch.equal(scale, scale_full))

        _, _, _, scale = scalar(
            torch.Tensor([[1e-10, 1e-10, 1e-10], [1e-15, 1e-15, 1e-15]]).reshape([2, 3, 1])
        )
        self.assertTrue(torch.equal(scale.flatten(), torch.Tensor([1e-10, 1.])))
