"""Offline regression checks for the data, forecast, and reporting contracts."""
import contextlib
import io
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
import torch

from main import parse_args, synthetic_data, data_fingerprint
from src.data import (_compound_window, _market_regression, load_data,
                      compute_characteristics, CHAR_NAMES, CACHE_VERSION, clean_price_history)
from src.models import PCAModel, IPCAModel, CAEModel
from src.evaluate import (factor_sharpe, get_portfolio_returns, nonlinear_contribution,
                          diebold_mariano_test, build_summary_table, _best_key_per_k,
                          format_paper_story, _bootstrap_months, bootstrap_sharpe_test,
                          run_significance_tests)
from src.ensemble import build_ensembles
from src import train


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.data = synthetic_data(17)
        cls.splits = cls.data['splits']
        cls.ret = cls.splits['train']['returns'].values.astype(np.float32)
        cls.chars = cls.splits['train']['chars']

    def test_missing_momentum_is_not_zero(self):
        values = _compound_window(np.array([[0.1, np.nan], [0.2, np.nan]]))
        self.assertAlmostEqual(values[0], 0.32)
        self.assertTrue(np.isnan(values[1]))

    def test_prices_exclude_security_break_and_nontrading_placeholders(self):
        dates = pd.to_datetime(['1993-09-30', '1993-10-01', '1993-10-04'])
        prices = pd.DataFrame({'NVR': [0.375, 10.25, 10.0], 'OTHER': [1., 10., 11.]}, index=dates)
        volume = pd.DataFrame({'NVR': [100., 100., 100.], 'OTHER': [0., 100., 100.]}, index=dates)
        clean = clean_price_history(prices, volume)
        returns = clean.pct_change(fill_method=None)
        self.assertTrue(returns.iloc[:2].isna().all().all())
        self.assertAlmostEqual(returns.loc['1993-10-04', 'NVR'], 10.0 / 10.25 - 1)
        self.assertAlmostEqual(returns.loc['1993-10-04', 'OTHER'], .1)
        self.assertEqual(prices.iloc[0, 0], .375)

    def test_ols_uses_paired_observations(self):
        market = np.linspace(-0.1, 0.1, 12)
        stocks = np.column_stack([0.01 + 2 * market, -0.02 - 3 * market])
        stocks[:4, 1] = np.nan
        beta, ivol = _market_regression(stocks, market)
        np.testing.assert_allclose(beta, [2, -3], atol=1e-12)
        np.testing.assert_allclose(ivol, 0, atol=1e-12)

    def test_nineteen_features_use_only_prior_month_information(self):
        rng = np.random.default_rng(33)
        dates = pd.date_range('1980-01-31', periods=90, freq='ME')
        columns = list('ABCDEFGH')
        returns = pd.DataFrame(rng.normal(0, 0.06, (90, 8)), index=dates, columns=columns)
        market = pd.Series(np.where(np.arange(90) % 2, -0.03, 0.04), index=dates)
        rf = pd.Series(0.001, index=dates)
        days = pd.bdate_range('1980-01-01', dates[-1])
        prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), 8)), axis=0)),
                              index=days, columns=columns)
        volume = pd.DataFrame(rng.uniform(1e5, 1e6, prices.shape), index=days, columns=columns)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            original = compute_characteristics(returns, market, rf, prices, volume)
            changed_returns = returns.copy()
            changed_returns.iloc[80:] *= 10
            changed_prices, changed_volume = prices.copy(), volume.copy()
            changed_prices.loc[changed_prices.index > dates[79]] *= 2
            changed_volume.loc[changed_volume.index > dates[79]] *= 4
            changed = compute_characteristics(changed_returns, market, rf, changed_prices, changed_volume)
        self.assertEqual(original.shape, (8, 90, 19))
        self.assertNotIn('log_mktcap', CHAR_NAMES)
        self.assertNotIn('turn1', CHAR_NAMES)
        self.assertTrue(np.isfinite(original[:, 70, :]).all())
        np.testing.assert_array_equal(original[:, :81], changed[:, :81])

    def test_data_pipeline_passes_raw_returns_once(self):
        raw = self.data['returns']
        rf = pd.Series(0.001, index=raw.index)
        market = pd.Series(0.01, index=raw.index)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch('src.data.get_sp500_tickers', return_value=list(raw.columns)), \
             patch('src.data.download_monthly_returns', return_value=(raw.abs() + 100, raw)), \
             patch('src.data.download_market_and_rf', return_value=(market, rf)), \
             patch('src.data.download_daily_data', return_value=(raw, raw)), \
             patch('src.data.compute_characteristics', return_value=self.data['chars']) as compute, \
             patch('src.data.MIN_MONTHS', 10):
            data = load_data(str(Path(tmp) / 'cache.pkl'))
        pd.testing.assert_frame_equal(compute.call_args.args[0], raw)
        np.testing.assert_allclose(data['returns'], raw - 0.001)
        self.assertEqual(data['cache_version'], CACHE_VERSION)

    def test_stale_cache_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.pkl'
            path.write_bytes(pickle.dumps({'chars': np.zeros((2, 2, 21))}))
            with self.assertRaisesRegex(ValueError, 'Stale data cache'):
                load_data(str(path))

    def test_pca_imputation_uses_training_means(self):
        model = PCAModel(2).fit(self.ret)
        missing = self.ret[:2].copy()
        missing[0, 0] = np.nan
        filled = np.where(np.isnan(missing), model.fill_values_, missing)
        np.testing.assert_allclose(model.get_factors(missing), model._pca.transform(filled))
        changed = missing.copy()
        changed[1, 0] += 10
        np.testing.assert_allclose(model.get_factors(missing)[0], model.get_factors(changed)[0])

    def test_ipca_final_factors_and_empty_months(self):
        ret = self.ret.copy()
        ret[0] = np.nan
        model = IPCAModel(2, max_iter=1).fit(ret, self.chars)
        factors = model.get_factors(ret, self.chars)
        self.assertTrue(np.isnan(factors[0]).all())
        np.testing.assert_allclose(model.factor_mean_, np.nanmean(factors, axis=0))
        np.testing.assert_allclose(model.factors_train_, factors)

    def test_cae_forecast_independent_of_realized_returns(self):
        model = CAEModel(n_chars=6, n_factors=2).fit()
        test_chars = self.splits['test']['chars']
        test = self.splits['test']['returns'].values
        pred = model.predict(test, test_chars, self.ret, self.chars)
        changed = model.predict(np.full_like(test, np.nan), test_chars, self.ret, self.chars)
        np.testing.assert_array_equal(pred, changed)
        bad_chars = test_chars.copy()
        bad_chars[0, 0, 0] = np.nan
        self.assertTrue(np.isnan(model.predict(test, bad_chars, self.ret, self.chars)[0, 0]))

    def test_cae_excludes_empty_factor_months(self):
        model = CAEModel(n_chars=6, n_factors=2).fit()
        ret = self.ret.copy()
        ret[0] = np.nan
        self.assertTrue(np.isnan(model.get_factors(ret, self.chars)[0]).all())

    def test_frozen_drift_is_measured_not_hardcoded(self):
        ipca = IPCAModel(2)
        ipca.gamma = np.ones((2, 6))
        model = CAEModel(n_chars=6, n_factors=2, freeze_linear=True).fit()
        with contextlib.redirect_stdout(io.StringIO()):
            model.initialize_from_ipca(self.ret, self.chars, ipca=ipca)
        with torch.no_grad():
            model.net.decoder.W_skip.weight.neg_()
        self.assertAlmostEqual(model.compute_ipca_drift(), 2.)

    def test_constant_betas_across_stocks_do_not_create_phi(self):
        model = CAEModel(n_chars=6, n_factors=2, use_linear=False).fit()
        with torch.no_grad():
            for param in model.net.parameters():
                param.zero_()
            model.net.decoder.g[-1].bias.copy_(torch.tensor([1., 2.]))
        self.assertTrue(np.isnan(nonlinear_contribution(model, self.ret, self.chars)[2]))

    def test_portfolio_sharpe_and_flat_signals_agree(self):
        rng = np.random.default_rng(99)
        ret = rng.normal(0, 0.1, (40, 30))
        pred = rng.normal(size=ret.shape)
        port = get_portfolio_returns(ret, pred)
        self.assertAlmostEqual(factor_sharpe(ret, pred), port.mean() / port.std(ddof=1) * np.sqrt(12))
        self.assertTrue(np.isnan(get_portfolio_returns(ret, np.ones_like(ret))).all())
        self.assertTrue(np.isnan(factor_sharpe(ret, np.ones_like(ret))))

    def test_dm_direction_identical_and_excess_lags(self):
        rng = np.random.default_rng(5)
        worse = rng.normal(0, 2, (50, 20))
        better = rng.normal(0, 0.2, (50, 20))
        result = diebold_mariano_test(worse, better, max_lag=100)
        self.assertGreater(result['dm_stat'], 0)
        self.assertEqual(result['n_lags'], 49)
        self.assertEqual(diebold_mariano_test(worse, worse)['p_value'], 1)

    def test_bootstrap_preserves_blocks_and_rejects_short_sharpe_series(self):
        indices = _bootstrap_months(np.random.default_rng(2), 120, 6)
        np.testing.assert_array_equal(np.diff(indices.reshape(-1, 6), axis=1) % 120, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = bootstrap_sharpe_test(np.arange(5), np.arange(5), n_bootstrap=10)
        self.assertTrue(np.isnan(result['p_value']))

    def test_selection_requires_scores_for_multiple_configs(self):
        configs = {(2, 0.1, 0.01): None, (2, 0.2, 0.01): None}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.pkl'
            with self.assertRaisesRegex(ValueError, 'Missing validation scores'):
                _best_key_per_k(configs, str(path))
            path.write_bytes(pickle.dumps({'pred_val_mse': {key: score for key, score in zip(configs, [2., 1.])}}))
            self.assertEqual(_best_key_per_k(configs, str(path))[2], (2, 0.2, 0.01))

    def test_labeled_significance_direction_and_holm(self):
        returns = self.splits['test']['returns'].values
        class PredictionStub:
            def __init__(self, fraction):
                self.prediction = fraction * returns
            def predict(self, *args, **kwargs):
                return self.prediction
        models = {family: {} for family in ('pca', 'ipca', 'cae', 'cae_fixed', 'cae_nl')}
        for k in (2, 3):
            models['pca'][k] = PredictionStub(0.0)
            models['ipca'][k] = PredictionStub(0.5)
            models['cae'][(k, 0.1, 0.01)] = PredictionStub(0.9)
            models['cae_fixed'][(k, 0.01)] = PredictionStub(0.7)
            models['cae_nl'][(k, 0.01)] = PredictionStub(0.2)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for family in ('cae', 'cae_fixed', 'cae_nl'):
                scores = {key: float(key[0]) for key in models[family]}
                (Path(tmp) / f'{family}_hparam_search.pkl').write_bytes(
                    pickle.dumps({'pred_val_mse': scores, 'best': min(scores, key=scores.get)}))
            results = run_significance_tests(self.splits, models, figures_dir=tmp,
                                               results_dir=tmp, n_bootstrap=10)
        for k in (2, 3):
            for key in ('dm_rescae_vs_ipca', 'dm_rescae_vs_pca', 'dm_rescae_vs_rescae_fixed'):
                test = results[k][key]
                self.assertGreater(test['dm_stat'], 0)
                self.assertGreaterEqual(test['p_value_holm'], test['p_value'])
            self.assertGreater(results['vs_cae_nl'][k]['dm_rescae_vs_cae_nl']['dm_stat'], 0)

    def test_summary_includes_ae_and_common_observations(self):
        data = self.splits.copy()
        data['test'] = dict(data['test'])
        data['test']['chars'] = data['test']['chars'].copy()
        data['test']['chars'][0, 0, :] = np.nan
        pca = PCAModel(2).fit(self.ret)
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_summary_table(data, {'pca': {2: pca}, 'ae': {2: pca}}, tmp)
        self.assertEqual(set(summary.Model), {'PCA', 'AE'})
        self.assertEqual(summary.N_predictions.nunique(), 1)
        self.assertEqual(summary.N_predictions.iloc[0], data['test']['returns'].size - 1)
        self.assertTrue(pd.api.types.is_numeric_dtype(summary.Pred_R2))

    def test_requested_k_device_and_fixed_seed_parity(self):
        # Actual optimization exercises the grid, seed wrappers, and frozen weights.
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(train, 'MAX_EPOCHS', 1), patch.object(train, 'PATIENCE', 1), \
             patch.object(train, 'LAMBDA_LIN_GRID', [0.001]), \
             patch.object(train, 'LAMBDA_NONLIN_GRID', [0.00001]):
            models = train.train_all_models(self.splits, k_list=[2], multi_seed=True,
                                             seeds=[5, 6], run_dir=tmp)
            for family in ('cae', 'cae_nl', 'cae_fixed'):
                self.assertEqual({key[0] for key in models[family]}, {2})
                fitted = next(iter(models[family].values()))
                self.assertEqual([m.seed for m in fitted], [5, 6])
                self.assertTrue(all(str(m.device) == 'cpu' for m in fitted))
                self.assertTrue(all(not hasattr(m, '_training_batches') for m in fitted))
            for model in next(iter(models['cae_fixed'].values())):
                self.assertEqual(model.compute_ipca_drift(), 0.)
                self.assertFalse(model.net.decoder.W_skip.weight.requires_grad)
            ensemble = build_ensembles(models)
            self.assertEqual(next(iter(ensemble['cae_fixed'].values())).n_seeds, 2)
            single = train.train_all_cae(self.splits, k_list=[2], device='cpu', run_dir=tmp,
                                         ipca_models=models['ipca'])
            self.assertTrue(next(iter(single.values())).use_linear is True)

    def test_story_uses_validation_k_and_fingerprint_changes(self):
        df = pd.DataFrame({'Model': ['ResCAE', 'ResCAE'], 'K': [2, 5], 'Pred_R2': [0.01, 0.5],
                           'Sharpe': [1., 2.], 'NL_frac': [0.2, 0.3], 'drift_rel': [0.1, 0.2]})
        story = format_paper_story(df, {'overall': {'rescae_key': (2, 0.1, 0.01)}, 'test_period': '2015–2024'})
        self.assertIn('(2, 0.1, 0.01)', story)
        self.assertIn('2015–2024', story)
        altered = {**self.data, 'splits': {**self.splits, 'test': {**self.splits['test'],
                    'returns': self.splits['test']['returns'] + 0.01}}}
        self.assertNotEqual(data_fingerprint(self.data), data_fingerprint(altered))

    def test_cli_does_not_silently_ignore_seeds(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--seeds', '1', '2'])


if __name__ == '__main__':
    unittest.main()
