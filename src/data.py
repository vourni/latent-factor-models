"""downloads and prepares data for the latent factor model study. sources: S&P 500 tickers from Wikipedia, prices/market from yfinance, risk-free from FRED."""

import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import rankdata, skew as _skew
from typing import Tuple

warnings.filterwarnings("ignore")

START_DATE = "1980-01-01"
END_DATE   = "2024-12-31"
MIN_MONTHS = 60          # minimum valid monthly observations to keep a stock
MAX_FILL   = 3           # max consecutive NaN months to forward-fill
TRAIN_END  = "2009-12-31"
VAL_END    = "2014-12-31"
# test window: 2015-01-01 → 2024-12-31


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
            raw = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            # Ensure DataFrame even for a single ticker
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(name=batch[0])
            monthly_px = raw.resample("ME").last()
            monthly_px_list.append(monthly_px)
        except Exception as e:
            print(f"  Batch {i}–{i+batch_size} failed: {e}")

    prices = pd.concat(monthly_px_list, axis=1)
    prices = prices.loc[~prices.index.duplicated(keep="first")]

    # Forward-fill short gaps, then compute returns
    prices_filled = prices.ffill(limit=MAX_FILL)
    returns = prices_filled.pct_change().iloc[1:]  # drop first NaN row

    # Drop stocks with too few observations
    valid_counts = returns.notna().sum(axis=0)
    keep = valid_counts[valid_counts >= MIN_MONTHS].index
    returns = returns[keep]

    # Align prices to the same dates and tickers as returns
    prices_out = prices_filled.loc[returns.index, keep]

    print(f"  Kept {len(keep)} stocks after quality filter.")
    return prices_out, returns


def download_market_and_rf() -> Tuple[pd.Series, pd.Series]:
    """downloads ^GSPC monthly returns and FRED TB3MS risk-free rate. returns (mkt_ret, rf)."""
    print("Downloading market index (^GSPC) …")
    gspc = yf.download("^GSPC", start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False)["Close"]
    # Newer yfinance may return a single-column DataFrame; squeeze to Series
    if isinstance(gspc, pd.DataFrame):
        gspc = gspc.squeeze()
    mkt_ret = gspc.resample("ME").last().pct_change().iloc[1:]
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


def download_shares_outstanding(tickers: list[str],
                                 cache_path: str = "data/raw/shares_cache.pkl",
                                 ) -> pd.DataFrame:
    """downloads monthly shares outstanding per ticker; falls back to current scalar if history unavailable. returns (T_months, N_tickers) df."""
    import os, pickle

    if os.path.exists(cache_path):
        print(f"Loading cached shares data from {cache_path} …")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    n = len(tickers)
    print(f"Downloading shares outstanding for {n} tickers …")
    all_months = pd.date_range(START_DATE, END_DATE, freq="ME")
    shares_dict: dict = {}
    batch_size = 50

    for i in range(0, n, batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  Downloading shares outstanding: {i}/{n}")

        for ticker in batch:
            try:
                tkr = yf.Ticker(ticker)
                shares_ts = tkr.get_shares_full(start=START_DATE)
                if shares_ts is not None and len(shares_ts) > 0:
                    # Resample to month-end, forward-fill gaps within the series,
                    # then reindex to the full monthly grid and forward-fill again.
                    s = shares_ts.resample("ME").last().ffill()
                    s = s.reindex(all_months, method="ffill")
                    shares_dict[ticker] = s
                else:
                    raise ValueError("empty series")
            except Exception:
                # Fall back to current scalar from .info; broadcast across all
                # dates.  Less accurate historically but unavoidable.
                try:
                    current = yf.Ticker(ticker).info.get("sharesOutstanding", None)
                    if current is not None:
                        shares_dict[ticker] = pd.Series(
                            float(current), index=all_months)
                    else:
                        shares_dict[ticker] = pd.Series(np.nan, index=all_months)
                except Exception:
                    shares_dict[ticker] = pd.Series(np.nan, index=all_months)

        if i + batch_size < n:
            time.sleep(0.5)

    print(f"  Downloading shares outstanding: {n}/{n}")
    df = pd.DataFrame(shares_dict, index=all_months)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)
    print(f"  Shares data cached to {cache_path}.")

    return df


def download_daily_data(tickers: list[str],
                        cache_path: str = "data/raw/daily_cache.pkl",
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
                end=END_DATE,
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


def compute_characteristics(returns: pd.DataFrame,
                             mkt_ret: pd.Series,
                             rf: pd.Series,
                             prices: pd.DataFrame = None,
                             shares: pd.DataFrame = None,
                             daily_prices: pd.DataFrame = None,
                             daily_volume: pd.DataFrame = None,
                             ) -> np.ndarray:
    """computes 21 lagged firm characteristics (momentum, vol, beta, mktcap, daily stats) and cross-sectionally rank-normalizes each to [-1, 1]. returns (N, T, 21) array."""
    T, N = returns.shape
    dates   = returns.index
    tickers = returns.columns

    mkt_ret = mkt_ret.reindex(dates)
    rf      = rf.reindex(dates).ffill()

    ret    = returns.values                  # (T, N)
    mkt    = np.asarray(mkt_ret).flatten()  # (T,)
    rf_arr = np.asarray(rf).flatten()       # (T,)

    chars_raw = np.full((T, N, 21), np.nan)

    # [6] mktcap: log(price * shares), both lagged one month
    if prices is not None and shares is not None:
        prices_aligned = prices.reindex(returns.index).reindex(
            columns=returns.columns)
        shares_aligned = shares.reindex(returns.index).reindex(
            columns=returns.columns)
        prices_lagged = prices_aligned.shift(1)
        shares_lagged = shares_aligned.shift(1)
        mktcap_raw = prices_lagged.values * shares_lagged.values
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mktcap = np.where(mktcap_raw > 1, np.log(mktcap_raw), np.nan)
        chars_raw[:, :, 6] = log_mktcap

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
        if shares is not None:
            shares_arr = (shares
                          .reindex(index=returns.index, columns=tickers)
                          .values)                   # (T, N)
        else:
            shares_arr = None
    else:
        daily_idx_arr = dp_arr = dv_arr = dr_arr = None
        monthly_dates_arr = shares_arr = None

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
            chars_raw[t, :, 1] = np.nanprod(1 + w6, axis=0) - 1

        # [2] mom12: cumulative return [t-13, t-2]
        w12 = ret[t - 13 : t - 1, :]
        if w12.shape[0] == 12:
            chars_raw[t, :, 2] = np.nanprod(1 + w12, axis=0) - 1

        # [3] vol12
        chars_raw[t, :, 3] = np.nanstd(ret[t - 12 : t, :], axis=0, ddof=1)

        # [4] beta12 + [5] ivol12: OLS market regression over 12 months
        ew12 = exc_ret[t - 12 : t, :]
        mw12 = mkt_exc[t - 12 : t]
        mv12 = np.nanvar(mw12, ddof=1)
        if mv12 > 0:
            cov12 = (np.nanmean(ew12 * mw12[:, None], axis=0)
                     - np.nanmean(ew12, axis=0) * np.nanmean(mw12))
            b12 = cov12 / mv12
            chars_raw[t, :, 4] = b12
            a12 = np.nanmean(ew12, axis=0) - b12 * np.nanmean(mw12)
            resid12 = ew12 - a12[None, :] - b12[None, :] * mw12[:, None]
            chars_raw[t, :, 5] = np.nanstd(resid12, axis=0, ddof=2)

        # [7] mom3: cumulative return [t-4, t-2]
        chars_raw[t, :, 7] = np.nanprod(1 + ret[t - 4 : t - 1, :], axis=0) - 1

        # [8] mom9: cumulative return [t-10, t-2]
        if t >= 10:
            chars_raw[t, :, 8] = (
                np.nanprod(1 + ret[t - 10 : t - 1, :], axis=0) - 1)

        # [9] mom36: cumulative return [t-37, t-13] (24-month window, t≥37)
        if t >= 37:
            chars_raw[t, :, 9] = (
                np.nanprod(1 + ret[t - 37 : t - 13, :], axis=0) - 1)

        # [11] vol6
        chars_raw[t, :, 11] = np.nanstd(ret[t - 6 : t, :], axis=0, ddof=1)

        # [12] beta36: OLS market beta over 36 months (t≥36)
        if t >= 36:
            ew36 = exc_ret[t - 36 : t, :]
            mw36 = mkt_exc[t - 36 : t]
            mv36 = np.nanvar(mw36, ddof=1)
            if mv36 > 0:
                cov36 = (np.nanmean(ew36 * mw36[:, None], axis=0)
                         - np.nanmean(ew36, axis=0) * np.nanmean(mw36))
                chars_raw[t, :, 12] = cov36 / mv36

        # [13] beta_down: downside beta, 60-month window, ≥12 negative mkt months
        if t >= 60:
            ew60  = exc_ret[t - 60 : t, :]
            mw60  = mkt_exc[t - 60 : t]
            dmask = mw60 < 0
            if dmask.sum() >= 12:
                mw_d = mw60[dmask]
                ew_d = ew60[dmask, :]
                mv_d = np.nanvar(mw_d, ddof=1)
                if mv_d > 0:
                    cov_d = (np.nanmean(ew_d * mw_d[:, None], axis=0)
                             - np.nanmean(ew_d, axis=0) * np.nanmean(mw_d))
                    chars_raw[t, :, 13] = cov_d / mv_d

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

        # [10] vol1: annualised daily vol × √21
        chars_raw[t, :, 10] = np.nanstd(dr_21, axis=0, ddof=1) * np.sqrt(21)

        # [14] max_ret / [15] min_ret over 21-day window
        with np.errstate(all="ignore"):
            chars_raw[t, :, 14] = np.nanmax(dr_21, axis=0)
            chars_raw[t, :, 15] = np.nanmin(dr_21, axis=0)

        # [16] high52: price / 52-week high (252 trading days including t1)
        if t1 >= 252:
            high_52w = np.nanmax(dp_arr[t1 - 251 : t1 + 1, :], axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                chars_raw[t, :, 16] = np.where(
                    high_52w > 0, dp_arr[t1, :] / high_52w, np.nan)

        # [17] skew1: 21-day skewness (unbiased, ≥5 obs)
        n21 = np.sum(~np.isnan(dr_21), axis=0)
        sk1 = _skew(dr_21, axis=0, bias=False, nan_policy="omit")
        chars_raw[t, :, 17] = np.where(n21 >= 5, sk1, np.nan)

        # [18] skew3: 63-day skewness (unbiased, ≥15 obs)
        t63s  = max(0, t1 - 62)
        dr_63 = dr_arr[t63s : t1 + 1, :]
        n63   = np.sum(~np.isnan(dr_63), axis=0)
        sk3   = _skew(dr_63, axis=0, bias=False, nan_policy="omit")
        chars_raw[t, :, 18] = np.where(n63 >= 15, sk3, np.nan)

        # [19] amihud: mean |r| / (px × vol) × 1e6, 252-day window, ≥60 valid
        if t1 >= 252:
            dr_252 = dr_arr[t1 - 251 : t1 + 1, :]
            dp_252 = dp_arr[t1 - 251 : t1 + 1, :]
            dv_252 = dv_arr[t1 - 251 : t1 + 1, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                dolvol = dp_252 * dv_252
                ratio  = np.where(dolvol > 0,
                                  np.abs(dr_252) / dolvol, np.nan)
            n_valid = np.sum(~np.isnan(ratio), axis=0)
            chars_raw[t, :, 19] = np.where(
                n_valid >= 60, np.nanmean(ratio, axis=0) * 1e6, np.nan)

        # [20] turn1: mean daily turnover (vol / shares) over 21-day window
        if shares_arr is not None:
            sh_tm1 = shares_arr[t - 1, :]   # (N,)
            with np.errstate(divide="ignore", invalid="ignore"):
                daily_turn = np.where(
                    sh_tm1[None, :] > 0, dv_21 / sh_tm1[None, :], np.nan)
            chars_raw[t, :, 20] = np.nanmean(daily_turn, axis=0)

    # cross-sectional rank normalization to [-1, 1] each month
    P_total    = chars_raw.shape[2]
    chars_norm = np.full_like(chars_raw, np.nan)
    for t in range(T):
        for p in range(P_total):
            x       = chars_raw[t, :, p]
            valid   = ~np.isnan(x)
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
        splits[name] = {
            "returns": returns.iloc[idx],        # (T_split, N)
            "chars":   chars[:, idx, :],         # (N, T_split, P)
        }
        T_s = mask.sum()
        print(f"  {name}: {T_s} months "
              f"({dates[mask][0].date()} → {dates[mask][-1].date()})")

    return splits


def load_data(cache_path: str = "data/raw/data_cache.pkl") -> dict:
    """runs the full data pipeline (tickers → returns → chars → splits) and caches the result. returns dict with keys 'splits', 'returns', 'chars', 'dates', 'tickers'."""
    import os, pickle

    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} …")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        cached_p = cached.get("chars", np.array([])).shape
        if len(cached_p) == 3 and cached_p[2] == 21:
            print(f"  Cache valid: P={cached_p[2]}, "
                  f"T={cached_p[1]}, N={cached_p[0]}.")
            return cached
        else:
            actual_p = cached_p[2] if len(cached_p) == 3 else "unknown"
            print(f"  WARNING: cached data has P={actual_p}, expected P=21. "
                  f"Rebuilding cache …")

    tickers = get_sp500_tickers()
    raw_prices, raw_returns = download_monthly_returns(tickers)
    mkt_ret, rf = download_market_and_rf()

    # Align all series to common date range and stock universe
    common_dates = (raw_returns.index
                    .intersection(rf.index)
                    .intersection(mkt_ret.index))
    raw_prices  = raw_prices.loc[common_dates]
    raw_returns = raw_returns.loc[common_dates]
    mkt_ret     = mkt_ret.loc[common_dates]
    rf          = rf.loc[common_dates]

    # Excess returns: subtract monthly risk-free rate from each stock
    excess_returns = raw_returns.subtract(rf, axis=0)

    print("Downloading shares outstanding …")
    shares = download_shares_outstanding(tickers)
    shares = shares.reindex(index=common_dates, columns=raw_returns.columns)

    print("Downloading daily prices and volume …")
    daily_prices, daily_volume = download_daily_data(tickers)
    # Restrict daily data to the stock universe that survived the quality filter
    daily_prices = daily_prices.reindex(columns=raw_returns.columns)
    daily_volume = daily_volume.reindex(columns=raw_returns.columns)

    print("Computing firm characteristics …")
    # Use excess_returns so momentum signals are consistent with the return
    # target — both strip out the risk-free rate, ensuring characteristics and
    # predicted returns are measured on the same basis.
    chars = compute_characteristics(excess_returns, mkt_ret, rf,
                                    prices=raw_prices, shares=shares,
                                    daily_prices=daily_prices,
                                    daily_volume=daily_volume)

    print("Splitting data …")
    splits = time_split(excess_returns, chars)

    result = {
        "splits":  splits,
        "returns": excess_returns,
        "chars":   chars,
        "dates":   common_dates,
        "tickers": list(raw_returns.columns),
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        import pickle as pk
        pk.dump(result, f)
    print(f"Data cached to {cache_path}.")

    return result
