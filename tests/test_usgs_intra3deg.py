"""Small scientific invariants for common-support spatial contrasts."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval.usgs_intra3deg_validation import pair_statistics, coarse_summary, bootstrap_summary


class IntraCellTests(unittest.TestCase):
    def test_common_months_remove_missing_in_any_model(self):
        obs = np.arange(36, dtype=float)
        pred = np.tile(obs, (5, 1)); pred[3, :13] = np.nan
        metrics, common, reason = pair_statistics(obs, pred, np.ones(36, bool), 24)
        self.assertIsNone(metrics); self.assertEqual(common.sum(), 23)
        self.assertEqual(reason, 'insufficient_common_months')
        metrics, common, _ = pair_statistics(obs, pred, np.ones(36, bool), 18)
        np.testing.assert_allclose(metrics[0], 1)

    def test_common_background_cancels_and_orientation_is_invariant(self):
        t = np.arange(36); obs = np.sin(t / 3)
        background = 100 * np.cos(t / 7)
        difference = (background + obs) - background
        pred = np.tile(difference, (5, 1))
        first, _, _ = pair_statistics(obs, pred, np.ones(36, bool), 24)
        reverse, _, _ = pair_statistics(-obs, -pred, np.ones(36, bool), 24)
        np.testing.assert_allclose(first, reverse)
        np.testing.assert_allclose(first[0], 1)

    def test_constant_prediction_excludes_every_model(self):
        obs = np.arange(36, dtype=float); pred = np.tile(obs, (5, 1)); pred[4] = 0
        metrics, _, reason = pair_statistics(obs, pred, np.ones(36, bool), 24)
        self.assertIsNone(metrics); self.assertIn('constant', reason)

    def test_sign_ties_are_literal(self):
        obs = np.tile([-1., 0., 1.], 12); pred = np.tile(obs, (5, 1)); pred[:, obs == 0] = .1
        metrics, _, _ = pair_statistics(obs, pred, np.ones(36, bool), 24)
        np.testing.assert_allclose(metrics[2], 2/3)

    def test_pair_density_does_not_weight_coarse_median(self):
        records = []
        for c, count, score in [(1, 100, -.5), (2, 1, .5)]:
            for i in range(count):
                for m in range(5):
                    records.append(dict(coarse_cell_id=c, model=f'M{m}', fine_cell_i=0,
                        fine_cell_j=i+1, pearson_r=score, spearman_rho=score, sign_agreement=.5))
        c = coarse_summary(pd.DataFrame(records)); rows = bootstrap_summary(c, 24, 'corrected')
        r = next(x for x in rows if x['comparison']=='M0' and x['metric']=='PCC')
        self.assertEqual(r['estimate'], 0)
        gain = next(x for x in rows if x['comparison']=='M4-M0' and x['metric']=='PCC' and x['statistic']=='median')
        self.assertEqual(gain['lower_95'], 0); self.assertEqual(gain['upper_95'], 0)
        self.assertEqual(rows, bootstrap_summary(c,24,'corrected'))


if __name__ == '__main__':
    unittest.main()
