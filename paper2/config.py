"""
All settings for the Paper 2 model in one place.
Change values here, not inside the other scripts.
"""

# ---------------------------------------------------------------- data
# Basket of assets (our name -> Yahoo Finance ticker). The FIRST one is the target coin.
BASKET = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SPX": "^GSPC",    # S&P 500 (US stocks)
    "GOLD": "GC=F",    # gold futures
    "US10Y": "^TNX",   # US 10-year Treasury yield, in percent
    "VIX": "^VIX",     # US stock "fear index" (the paper uses VIX futures)
}
TARGET = "BTC"           # the coin we want the regime for
CALENDAR = "SPX"         # align every asset on the days this market is open
YIELD_COLS = ["US10Y"]   # yields: use the daily change in level, not a log return
START = "2017-11-10"     # ETH history on Yahoo starts in Nov 2017

# ---------------------------------------------------------------- model
WINDOW = 21              # business days in the rolling covariance (about 1 month)
MIN_TRAIN = 504          # business days of history before the first fit (about 2 years)
REFIT_EVERY = 21         # refit the model once a month
N_PCS = 3                # principal components kept (same as the paper)
N_REGIMES = 2            # calm vs. highly volatile (same as the paper)
FEATURE_SET = "adapted"  # "adapted": volatilities + correlations (our version)
                         # "paper":   correlation distances only (closest to the paper)
MIN_STAY = 5             # a new regime must last this many days before the label switches

# ---------------------------------------------------------------- backtest
HORIZON = 20             # days ahead for the "what happened next" checks
FAST_MA, SLOW_MA = 30, 100   # the paper's simple trading rule: 30-day vs 100-day average
COST = 0.001             # 0.1% cost per trade
DAYS_PER_YEAR = 252      # business days (weekend crypto moves land in Monday's return)

# Well-known stress periods, used only to sanity-check real-data results (approximate dates)
STRESS_WINDOWS = {
    "COVID crash (Mar 2020)": ("2020-03-09", "2020-03-31"),
    "May 2021 crypto crash": ("2021-05-12", "2021-06-30"),
    "Terra/LUNA (May-Jun 2022)": ("2022-05-05", "2022-06-30"),
    "FTX collapse (Nov 2022)": ("2022-11-07", "2022-11-30"),
}
