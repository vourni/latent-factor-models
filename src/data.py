"""downloads and prepares data for the latent factor model study. sources: S&P 500 tickers from Wikipedia, prices/market from yfinance, risk-free from FRED."""

import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import rankdata, skew as _skew
from typing import Tuple

START_DATE = "1980-01-01"
END_DATE   = "2024-12-31"
DOWNLOAD_END = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
CACHE_VERSION = 4
# Do not join predecessor units to reorganized common shares. NVR's filing:
# https://www.sec.gov/Archives/edgar/data/906163/000095012310100067/w79957e10vq.htm
PRICE_HISTORY_STARTS = {"NVR": "1993-10-01"}
CHAR_NAMES = [
    "mom1", "mom6", "mom12", "vol12", "beta12", "ivol12",
    "mom3", "mom9", "mom24_13", "vol1", "vol6", "beta36", "beta_down",
    "max_ret", "min_ret", "high52", "skew1", "skew3", "amihud",
]
MIN_MONTHS = 60          # minimum valid TRAINING observations to keep a stock
MAX_FILL   = 3           # max consecutive NaN months to forward-fill
TRAIN_END  = "2009-12-31"
VAL_END    = "2014-12-31"
# test window: 2015-01-01 → 2024-12-31


def clean_price_history(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Exclude non-trading placeholders and a known security-identity boundary."""
    cleaned = prices.where(volume.reindex_like(prices) > 0).copy()
    for ticker, start in PRICE_HISTORY_STARTS.items():
        if ticker in cleaned:
            cleaned.loc[cleaned.index < start, ticker] = np.nan
    return cleaned


# survivorship bias note: this uses the *current* S&P 500 list, so delisted names are missing
def get_sp500_tickers() -> list[str]:
    """scrapes current S&P 500 tickers from Wikipedia. returns list of ticker strings."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(pd.io.common.StringIO(resp.text), flavor="lxml")
    tickers = tables[0]["Symbol"].tolist()
    # Wikipedia uses dots (e.g. BRK.B); yfinance expects dashes (BRK-B)
    return [t.replace(".", "-") for t in tickers]


def download_monthly_returns(tickers: list[str],
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """downloads monthly prices and returns for given tickers. returns (prices df, returns df)."""
    print(f"Downloading prices for {len(tickers)} tickers …")
    batch_size = 100
    monthly_px_list = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            downloaded = yf.download(
                batch,
                start=START_DATE,
                end=DOWNLOAD_END,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            raw, volume = downloaded["Close"], downloaded["Volume"]
            # Ensure DataFrame even for a single ticker
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(name=batch[0])
                volume = volume.to_frame(name=batch[0])
            raw = clean_price_history(raw, volume)
            monthly_px = raw.resample("ME").last()
            monthly_px_list.append(monthly_px)
        except Exception as e:
            print(f"  Batch {i}–{i+batch_size} failed: {e}")

    if not monthly_px_list:
        raise RuntimeError("No price batches were downloaded successfully.")
    prices = pd.concat(monthly_px_list, axis=1)
    prices = prices.loc[~prices.index.duplicated(keep="first")]

    # Forward-fill short gaps, then compute returns
    prices_filled = prices.ffill(limit=MAX_FILL)
    returns = prices_filled.pct_change(fill_method=None).iloc[1:]

    # Drop stocks with too few observations
    valid_counts = returns.loc[:TRAIN_END].notna().sum(axis=0)
    keep = valid_counts[valid_counts >= MIN_MONTHS].index
    returns = returns[keep]

    # Align prices to the same dates and tickers as returns
    prices_out = prices_filled.loc[returns.index, keep]

    print(f"  Kept {len(keep)} stocks after quality filter.")
    return prices_out, returns


def download_market_and_rf() -> Tuple[pd.Series, pd.Series]:
    """downloads ^GSPC monthly returns and FRED TB3MS risk-free rate. returns (mkt_ret, rf)."""
    print("Downloading market index (^GSPC) …")
    gspc = yf.download("^GSPC", start=START_DATE, end=DOWNLOAD_END,
                       auto_adjust=True, progress=False)["Close"]
    # Newer yfinance may return a single-column DataFrame; squeeze to Series
    if isinstance(gspc, pd.DataFrame):
        gspc = gspc.squeeze()
    mkt_ret = gspc.resample("ME").last().pct_change(fill_method=None).iloc[1:]
    mkt_ret.name = "mkt"

    print("Downloading risk-free rate from FRED (TB3MS) …")
    # FRED exposes a public CSV endpoint that requires no API key
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"
    resp = requests.get(fred_url, timeout=30)
    resp.raise_for_status()
    tb3ms = pd.read_csv(
        pd.io.common.StringIO(resp.text),
        index_col=0, parse_dates=True,
    ).squeeze()
    tb3ms.index = pd.to_datetime(tb3ms.index)
    # Annualized % → monthly decimal
    rf = (1 + tb3ms / 100) ** (1 / 12) - 1
    rf = rf.resample("ME").last()
    rf.name = "rf"

    return mkt_ret, rf


def download_daily_data(tickers: list[str],
                        cache_path: str = "data/raw/daily_cache_v3.pkl",
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """downloads daily adjusted-close and volume; used for vol1, max_ret, amihud, etc. returns (daily_prices, daily_volume)."""
    import os, pickle

    if os.path.exists(cache_path):
        print(f"Loading cached daily data from {cache_path} …")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    n = len(tickers)
    print(f"Downloading daily prices and volume for {n} tickers …")
    batch_size = 100
    price_list: list = []
    volume_list: list = []

    for i in range(0, n, batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                end=DOWNLOAD_END,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            px  = raw["Close"]
            vol = raw["Volume"]
            if isinstance(px, pd.Series):
                px  = px.to_frame(name=batch[0])
            if isinstance(vol, pd.Series):
                vol = vol.to_frame(name=batch[0])
            price_list.append(px)
            volume_list.append(vol)
        except Exception as e:
            print(f"  Daily batch {i}–{i + batch_size} failed: {e}")

    if not price_list:
        raise RuntimeError("No daily price batches were downloaded successfully.")
    daily_prices = pd.concat(price_list,  axis=1)
    daily_volume = pd.concat(volume_list, axis=1)
    daily_prices = daily_prices.loc[:, ~daily_prices.columns.duplicated()]
    daily_volume = daily_volume.loc[:, ~daily_volume.columns.duplicated()]

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((daily_prices, daily_volume), f)
    print(f"  Daily data cached to {cache_path}.")

    return daily_prices, daily_volume


def _compound_window(window: np.ndarray) -> np.ndarray:
    """Compound complete return windows; missing history is not a zero return."""
    return np.where(np.isfinite(window).all(axis=0),
                    np.prod(1 + window, axis=0) - 1, np.nan)


def _market_regression(stock: np.ndarray, market: np.ndarray):
    """Per-stock OLS with intercept, using paired finite observations."""
    valid = np.isfinite(stock) & np.isfinite(market[:, None])
    count = valid.sum(axis=0)
    denom = np.maximum(count, 1)
    x_mean = np.where(valid, market[:, None], 0).sum(axis=0) / denom
    y_mean = np.where(valid, stock, 0).sum(axis=0) / denom
    dx = np.where(valid, market[:, None] - x_mean, 0)
    dy = np.where(valid, stock - y_mean, 0)
    ss_x = (dx ** 2).sum(axis=0)
    beta = np.divide((dx * dy).sum(axis=0), ss_x,
                     out=np.full(stock.shape[1], np.nan), where=(count >= 3) & (ss_x > 0))
    residual = np.where(valid, dy - beta * dx, 0)
    ivol = np.where(count >= 3,
                    np.sqrt((residual ** 2).sum(axis=0) / np.maximum(count - 2, 1)), np.nan)
    return beta, ivol


def compute_characteristics(returns: pd.DataFrame,
                             mkt_ret: pd.Series,
                             rf: pd.Series,
                             daily_prices: pd.DataFrame = None,
                             daily_volume: pd.DataFrame = None,
                             ) -> np.ndarray:
    """Compute 19 lagged characteristics from RAW returns, ranked to [-1, 1].
    Risk-free returns are subtracted here only for market regressions.
    Returns an (N, T, 19) array.
    """
    T, N = returns.shape
    dates   = returns.index
    tickers = returns.columns

    mkt_ret = mkt_ret.reindex(dates)
    rf      = rf.reindex(dates).ffill()

    ret    = returns.values                  # (T, N)
    mkt    = np.asarray(mkt_ret).flatten()  # (T,)
    rf_arr = np.asarray(rf).flatten()       # (T,)

    chars_raw = np.full((T, N, len(CHAR_NAMES)), np.nan)

    # pre-process daily data into arrays for fast monthly indexing
    have_daily = daily_prices is not None and daily_volume is not None
    if have_daily:
        dp = daily_prices.reindex(columns=tickers).ffill(limit=5)
        dv = daily_volume.reindex(columns=tickers).fillna(0.0)

        daily_idx_arr     = dp.index.values          # (T_days,) datetime64
        dp_arr            = dp.values.astype(float)  # (T_days, N)
        dv_arr            = dv.values.astype(float)  # (T_days, N)

        # dr_arr[i] = daily return on day daily_idx_arr[i]; dr_arr[0] = NaN
        dr_arr = np.full_like(dp_arr, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            dr_arr[1:] = (dp_arr[1:]
                          / np.where(dp_arr[:-1] > 0, dp_arr[:-1], np.nan)
                          - 1)

        monthly_dates_arr = dates.values             # (T,) datetime64
    else:
        daily_idx_arr = dp_arr = dv_arr = dr_arr = None
        monthly_dates_arr = None

    # excess returns used for beta/ivol calculations
    exc_ret = ret - rf_arr[:, None]   # (T, N)
    mkt_exc = mkt - rf_arr            # (T,)

    # main monthly loop — compute each characteristic at t using data through t-1
    for t in range(13, T):

        # [0] mom1
        chars_raw[t, :, 0] = ret[t - 1, :]

        # [1] mom6: cumulative return [t-7, t-2]
        w6 = ret[t - 7 : t - 1, :]
        if w6.shape[0] == 6:
            chars_raw[t, :, 1] = _compound_window(w6)

        # [2] mom12: cumulative return [t-13, t-2]
        w12 = ret[t - 13 : t - 1, :]
        if w12.shape[0] == 12:
            chars_raw[t, :, 2] = _compound_window(w12)

        # [3] vol12
        chars_raw[t, :, 3] = np.nanstd(ret[t - 12 : t, :], axis=0, ddof=1)

        # [4] beta12 + [5] ivol12: OLS market regression over 12 months
        ew12 = exc_ret[t - 12 : t, :]
        mw12 = mkt_exc[t - 12 : t]
        chars_raw[t, :, 4], chars_raw[t, :, 5] = _market_regression(ew12, mw12)

        # [6] mom3: cumulative return [t-4, t-2]
        chars_raw[t, :, 6] = _compound_window(ret[t - 4 : t - 1, :])

        # [7] mom9: cumulative return [t-10, t-2]
        if t >= 10:
            chars_raw[t, :, 7] = (
                _compound_window(ret[t - 10 : t - 1, :]))

        # [8] mom24_13: half-open [t-37, t-13), 24 months with a 13-month gap
        if t >= 37:
            chars_raw[t, :, 8] = (
                _compound_window(ret[t - 37 : t - 13, :]))

        # [10] vol6
        chars_raw[t, :, 10] = np.nanstd(ret[t - 6 : t, :], axis=0, ddof=1)

        # [11] beta36: OLS market beta over 36 months (t≥36)
        if t >= 36:
            ew36 = exc_ret[t - 36 : t, :]
            mw36 = mkt_exc[t - 36 : t]
            chars_raw[t, :, 11], _ = _market_regression(ew36, mw36)

        # [12] beta_down: downside beta, 60-month window, ≥12 negative mkt months
        if t >= 60:
            ew60  = exc_ret[t - 60 : t, :]
            mw60  = mkt_exc[t - 60 : t]
            dmask = mw60 < 0
            if dmask.sum() >= 12:
                mw_d = mw60[dmask]
                ew_d = ew60[dmask, :]
                beta_d, _ = _market_regression(ew_d, mw_d)
                beta_d[np.isfinite(ew_d).sum(axis=0) < 12] = np.nan
                chars_raw[t, :, 12] = beta_d

        # daily characteristics (skipped if daily data not provided)
        if not have_daily:
            continue

        # Last daily observation on or before end of month t-1
        month_end_tm1 = monthly_dates_arr[t - 1]
        t1 = int(np.searchsorted(daily_idx_arr, month_end_tm1, side="right")) - 1
        if t1 < 1:
            continue

        # 21-day daily window ending at t1 (inclusive)
        t21s  = max(0, t1 - 20)
        dr_21 = dr_arr[t21s : t1 + 1, :]   # (≤21, N)
        dv_21 = dv_arr[t21s : t1 + 1, :]   # (≤21, N)

        # [9] vol1: daily volatility scaled to a 21-trading-day month
        chars_raw[t, :, 9] = np.nanstd(dr_21, axis=0, ddof=1) * np.sqrt(21)

        # [13] max_ret / [14] min_ret over 21-day window
        with np.errstate(all="ignore"):
            chars_raw[t, :, 13] = np.nanmax(dr_21, axis=0)
            chars_raw[t, :, 14] = np.nanmin(dr_21, axis=0)

        # [15] high52: price / 52-week high (252 trading days including t1)
        if t1 >= 252:
            high_52w = np.nanmax(dp_arr[t1 - 251 : t1 + 1, :], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                chars_raw[t, :, 15] = np.where(
                    high_52w > 0, dp_arr[t1, :] / high_52w, np.nan)

        # [16] skew1: 21-day skewness (unbiased, ≥5 obs)
        n21 = np.sum(~np.isnan(dr_21), axis=0)
        sk1 = _skew(dr_21, axis=0, bias=False, nan_policy="omit")
        chars_raw[t, :, 16] = np.where(n21 >= 5, sk1, np.nan)

        # [17] skew3: 63-day skewness (unbiased, ≥15 obs)
        t63s  = max(0, t1 - 62)
        dr_63 = dr_arr[t63s : t1 + 1, :]
        n63   = np.sum(~np.isnan(dr_63), axis=0)
        sk3   = _skew(dr_63, axis=0, bias=False, nan_policy="omit")
        chars_raw[t, :, 17] = np.where(n63 >= 15, sk3, np.nan)

        # [18] amihud: mean |r| / (px × vol) × 1e6, 252-day window, ≥60 valid
        if t1 >= 252:
            dr_252 = dr_arr[t1 - 251 : t1 + 1, :]
            dp_252 = dp_arr[t1 - 251 : t1 + 1, :]
            dv_252 = dv_arr[t1 - 251 : t1 + 1, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                dolvol = dp_252 * dv_252
                ratio  = np.where(dolvol > 0,
                                  np.abs(dr_252) / dolvol, np.nan)
            n_valid = np.sum(~np.isnan(ratio), axis=0)
            chars_raw[t, :, 18] = np.where(
                n_valid >= 60, np.nanmean(ratio, axis=0) * 1e6, np.nan)

    # cross-sectional rank normalization to [-1, 1] each month
    P_total    = chars_raw.shape[2]
    chars_norm = np.full_like(chars_raw, np.nan)
    for t in range(T):
        for p in range(P_total):
            x       = chars_raw[t, :, p]
            valid   = np.isfinite(x)
            n_valid = valid.sum()
            if n_valid < 2:
                continue
            ranks        = np.full(N, np.nan)
            ranks[valid] = rankdata(x[valid], method="average")
            chars_norm[t, valid, p] = (
                2 * (ranks[valid] - 1) / (n_valid - 1) - 1)

    chars_out = np.transpose(chars_norm, (1, 0, 2))
    assert chars_out.shape[2] == P_total, \
        f"Expected P={P_total}, got {chars_out.shape[2]}"
    return chars_out


def time_split(returns: pd.DataFrame,
               chars: np.ndarray
               ) -> dict:
    """splits returns and chars into train/val/test by calendar date; no shuffling. returns dict with keys 'train', 'val', 'test'."""
    dates = returns.index
    train_mask = dates <= TRAIN_END
    val_mask   = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask  = dates > VAL_END

    splits = {}
    for name, mask in [("train", train_mask),
                        ("val",   val_mask),
                        ("test",  test_mask)]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError(f"The {name} split is empty; check the data date range.")
        splits[name] = {
            "returns": returns.iloc[idx],        # (T_split, N)
            "chars":   chars[:, idx, :],         # (N, T_split, P)
        }
        T_s = mask.sum()
        print(f"  {name}: {T_s} months "
              f"({dates[mask][0].date()} → {dates[mask][-1].date()})")

    return splits


def load_data(cache_path: str = "data/raw/data_cache_v4.pkl") -> dict:
    """Build a 19-feature panel with common model eligibility and train-only filters."""
    import os, pickle
    from datetime import datetime, timezone

    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} …")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if (cached.get("cache_version") != CACHE_VERSION
                or cached.get("characteristic_names") != CHAR_NAMES):
            raise ValueError("Stale data cache. Use a new --cache path to rebuild the current feature panel.")
        return cached

    tickers = get_sp500_tickers()
    raw_prices, raw_returns = download_monthly_returns(tickers)
    mkt_ret, rf = download_market_and_rf()
    common_dates = raw_returns.index.intersection(rf.index).intersection(mkt_ret.index)
    raw_returns = raw_returns.loc[common_dates]
    mkt_ret, rf = mkt_ret.loc[common_dates], rf.loc[common_dates]
    excess_returns = raw_returns.subtract(rf, axis=0)

    daily_prices, daily_volume = download_daily_data(
        list(raw_returns.columns), os.path.join(os.path.dirname(cache_path), "daily_cache_v3.pkl"))
    daily_prices = daily_prices.reindex(columns=raw_returns.columns)
    daily_volume = daily_volume.reindex(columns=raw_returns.columns)
    daily_prices = clean_price_history(daily_prices, daily_volume)
    print("Computing 19 lagged price/volume characteristics …")
    chars = compute_characteristics(raw_returns, mkt_ret, rf,
                                    daily_prices=daily_prices, daily_volume=daily_volume)
    eligible = np.isfinite(excess_returns.values) & np.isfinite(chars).all(axis=2).T
    # Every model uses exactly the same observed training/validation/test panel.
    masked_returns = excess_returns.where(eligible)
    keep_stocks = masked_returns.loc[:TRAIN_END].notna().sum(axis=0) >= MIN_MONTHS
    masked_returns = masked_returns.loc[:, keep_stocks]
    chars = chars[keep_stocks.values]
    keep_months = masked_returns.notna().any(axis=1)
    masked_returns = masked_returns.loc[keep_months]
    chars = chars[:, keep_months.values, :]
    if masked_returns.empty:
        raise ValueError("No eligible observations after the training-coverage filter.")
    print(f"Common panel: {masked_returns.shape[1]} stocks, {len(masked_returns)} months.")
    splits = time_split(masked_returns, chars)
    result = {
        "splits": splits, "returns": masked_returns, "chars": chars,
        "dates": masked_returns.index, "tickers": list(masked_returns.columns),
        "cache_version": CACHE_VERSION, "characteristic_names": CHAR_NAMES,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "universe_snapshot": tickers,
        "source_dates": {"start": START_DATE, "end": END_DATE},
        "training_min_observations": MIN_MONTHS,
        "price_history_starts": PRICE_HISTORY_STARTS,
        "price_quality_rules": ["Exclude quotes with nonpositive volume before resampling/filling",
                                "NVR price history starts after the 1993 reorganization"],
        "coverage": {name: {"months": len(part["returns"]),
                            "observed_pairs": int(part["returns"].notna().sum().sum())}
                     for name, part in splits.items()},
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Data cached to {cache_path}.")
    return result
