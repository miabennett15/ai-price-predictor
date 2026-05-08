"""
Currency Price Predictor  ─  LSTM + 9-Panel Diagnostic Dashboard
=================================================================
Downloads live FX/crypto data, trains an LSTM, predicts future prices,
validates accuracy, and saves a comprehensive visual report.

Requirements:
    pip install yfinance numpy pandas scikit-learn tensorflow matplotlib seaborn scipy

Usage:
    python currency_predictor.py                   # EUR/USD, 7-day forecast
    python currency_predictor.py --pair GBPUSD=X   # GBP/USD forex
    python currency_predictor.py --pair BTC-USD    # Bitcoin
    python currency_predictor.py --pair GBPUSD=X --days 14 --period 3y
"""

import argparse
import warnings
import sys

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ── Lazy imports ───────────────────────────────────────────────────────────────
def _require(package, pip_name=None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        pip_name = pip_name or package
        sys.exit(f"[ERROR] '{pip_name}' not installed.  Run:  pip install {pip_name}")

yf = _require("yfinance")
_require("sklearn.preprocessing", "scikit-learn")
_require("sklearn.metrics", "scikit-learn")
_require("tensorflow")
_require("scipy")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import gaussian_kde, norm


# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    pair="EURUSD=X",
    period="5y",
    seq_len=60,
    future_days=7,
    test_split=0.15,
    epochs=100,
    batch_size=32,
)

# ── Design tokens ──────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEAL   = "#39d0a0"
CORAL  = "#ff6b6b"
AMBER  = "#fbbf24"
VIOLET = "#a78bfa"
SLATE  = "#94a3b8"
WHITE  = "#f0f6fc"
DIMMED = "#6e7681"
GREEN  = "#3fb950"
RED    = "#f85149"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_data(pair: str, period: str) -> pd.DataFrame:
    print(f"\n[1/5] Fetching  {pair}  ({period}) ...")
    ticker = yf.Ticker(pair)
    df = ticker.history(period=period)
    if df.empty:
        sys.exit(f"[ERROR] No data returned for '{pair}'. Check the ticker symbol.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 200:
        sys.exit(f"[ERROR] Only {len(df)} rows — need at least 200 data points.")
    print(f"         {len(df)} trading days  |  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  +  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c  = df["Close"].copy()
    df = df.copy()
    df["ret"]      = c.pct_change()
    df["sma_20"]   = c.rolling(20).mean()
    df["sma_50"]   = c.rolling(50).mean()
    df["ema_12"]   = c.ewm(span=12).mean()
    df["ema_26"]   = c.ewm(span=26).mean()
    df["macd"]     = df["ema_12"] - df["ema_26"]
    df["vol_20"]   = c.rolling(20).std()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["bb_upper"] = df["sma_20"] + 2 * df["vol_20"]
    df["bb_lower"] = df["sma_20"] - 2 * df["vol_20"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["sma_20"] + 1e-9)
    return df.dropna()


FEAT_COLS = ["Close", "ret", "sma_20", "vol_20", "rsi", "macd", "bb_width"]


def build_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, 0])   # predict Close (column 0)
    return np.array(X), np.array(y)


def preprocess(df: pd.DataFrame, seq_len: int, test_split: float):
    print("\n[2/5] Preprocessing ...")
    feat_df   = add_features(df)
    feat_data = feat_df[FEAT_COLS].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feat_data)

    X, y   = build_sequences(scaled, seq_len)
    split  = int(len(X) * (1 - test_split))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    close_scaler = MinMaxScaler()
    close_scaler.fit(feat_df[["Close"]].values)

    test_dates = feat_df.index[seq_len + split:]
    print(f"         Features: {len(FEAT_COLS)}  |  Train: {len(X_tr):,}  "
          f"|  Test: {len(X_te):,}  |  Seq-len: {seq_len}")
    return X_tr, X_te, y_tr, y_te, scaler, close_scaler, scaled, feat_df, test_dates


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(seq_len: int, n_feat: int) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, n_feat)),
        Dropout(0.2),
        LSTM(64,  return_sequences=True),
        Dropout(0.2),
        LSTM(32,  return_sequences=False),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")
    return model


def train_model(model, X_tr, y_tr, epochs, batch_size):
    print("\n[3/5] Training LSTM ...")
    cb = [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=6, verbose=0),
    ]
    history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, callbacks=cb, verbose=1)
    print(f"         Stopped at epoch {len(history.epoch)}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate(model, X_te, y_te, close_scaler, pair):
    print("\n[4/5] Validating ...")
    y_pred_s = model.predict(X_te, verbose=0).flatten()
    y_actual = close_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    y_pred   = close_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-9))) * 100
    r2   = r2_score(y_actual, y_pred)
    d_act = np.sign(np.diff(y_actual))
    d_prd = np.sign(np.diff(y_pred))
    dir_acc = np.mean(d_act == d_prd) * 100

    metrics = dict(MAE=mae, RMSE=rmse, MAPE_pct=mape, R2=r2,
                   Directional_Accuracy_pct=dir_acc)

    print(f"  MAE {mae:.4f}  |  RMSE {rmse:.4f}  |  MAPE {mape:.2f}%  "
          f"|  R2 {r2:.4f}  |  Dir-Acc {dir_acc:.1f}%")
    return y_actual, y_pred, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FUTURE FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def forecast_future(model, scaled, close_scaler, seq_len, future_days, n_feat):
    """Autoregressively step forward `future_days` beyond last known bar."""
    window  = list(scaled[-seq_len:])
    preds_s = []
    for _ in range(future_days):
        x   = np.array(window[-seq_len:]).reshape(1, seq_len, n_feat)
        nxt = model.predict(x, verbose=0)[0][0]
        new_row    = window[-1].copy()
        new_row[0] = nxt
        preds_s.append(nxt)
        window.append(new_row)

    preds      = close_scaler.inverse_transform(
                     np.array(preds_s).reshape(-1, 1)).flatten()
    last_price = close_scaler.inverse_transform([[scaled[-1, 0]]])[0][0]

    print(f"\n[5/5] {future_days}-day Forecast  (last known: {last_price:.4f})")
    for i, p in enumerate(preds, 1):
        arrow = "^" if p >= last_price else "v"
        print(f"  +{i:>2d}: {p:>10.4f}  {arrow} {(p - last_price) / last_price * 100:+.2f}%")
    return preds.tolist(), last_price


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTS  —  9-panel diagnostic dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SLATE, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    ax.title.set_color(WHITE)
    ax.grid(color=BORDER, linewidth=0.35, linestyle="--", alpha=0.55)
    return ax


def plot_results(feat_df, test_dates, y_actual, y_pred,
                 future_preds, last_price, metrics, history,
                 pair, future_days, seq_len, outfile="prediction_report.png"):

    print("\n[6/5] Generating 9-panel diagnostic report ...")

    prices    = feat_df["Close"]
    residuals = y_actual - y_pred
    roll_win  = max(5, len(residuals) // 20)

    # ── Derived series ─────────────────────────────────────────────────────────
    roll_mae = pd.Series(np.abs(residuals)).rolling(roll_win).mean().values

    dir_actual  = np.sign(np.diff(y_actual))
    dir_pred_s  = np.sign(np.diff(y_pred))
    dir_correct = (dir_actual == dir_pred_s).astype(int)
    roll_dir    = pd.Series(dir_correct).rolling(roll_win).mean().values * 100

    ret_actual   = np.diff(y_actual) / (y_actual[:-1] + 1e-9) * 100
    strategy_ret = dir_pred_s * ret_actual
    cum_strategy = np.cumprod(1 + strategy_ret / 100) - 1
    cum_bh       = np.cumprod(1 + ret_actual   / 100) - 1

    roll_std   = pd.Series(residuals).rolling(roll_win, min_periods=1).std().values
    upper_band = y_pred + roll_std
    lower_band = y_pred - roll_std

    fut_dates  = pd.bdate_range(start=prices.index[-1], periods=future_days + 1)[1:]
    sigma_last = np.std(residuals[-30:]) if len(residuals) >= 30 else np.std(residuals)
    cone_upper = [last_price + sigma_last * np.sqrt(i + 1) for i in range(future_days)]
    cone_lower = [last_price - sigma_last * np.sqrt(i + 1) for i in range(future_days)]

    # ── Figure layout ──────────────────────────────────────────────────────────
    #   Row 0  : title / metrics banner  (thin)
    #   Row 1  : Panel A  – full-width price chart
    #   Row 2  : Panels B · C · D
    #   Row 3  : Panels E · F · G
    fig = plt.figure(figsize=(24, 20), facecolor=BG)
    gs  = gridspec.GridSpec(
        4, 3,
        figure=fig,
        height_ratios=[0.07, 1.1, 0.9, 0.9],
        hspace=0.55, wspace=0.38,
        left=0.06, right=0.97, top=0.97, bottom=0.05,
    )

    # ── Title / metrics banner ─────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.set_facecolor(BG)
    ax_t.axis("off")
    ax_t.text(0.0, 0.6, "LSTM Currency Prediction Report",
              fontsize=18, color=WHITE, fontweight="bold", va="center",
              fontfamily="monospace", transform=ax_t.transAxes)
    ax_t.text(1.0, 0.6,
              f"{pair}  |  {future_days}-day forecast  |  look-back {seq_len}d",
              fontsize=9, color=DIMMED, va="center", ha="right",
              fontfamily="monospace", transform=ax_t.transAxes)

    metric_items = [
        ("MAE",       f"{metrics['MAE']:.4f}",                      TEAL),
        ("RMSE",      f"{metrics['RMSE']:.4f}",                     TEAL),
        ("MAPE",      f"{metrics['MAPE_pct']:.2f}%",                AMBER),
        ("R2",        f"{metrics['R2']:.4f}",                       TEAL),
        ("Dir. Acc.", f"{metrics['Directional_Accuracy_pct']:.1f}%",
         GREEN if metrics["Directional_Accuracy_pct"] >= 50 else RED),
    ]
    for x, (lbl, val, col) in zip(
            np.linspace(0.20, 0.96, len(metric_items)), metric_items):
        ax_t.text(x, 0.88, lbl, ha="center", va="top", fontsize=7,
                  color=DIMMED, fontfamily="monospace", transform=ax_t.transAxes)
        ax_t.text(x, 0.40, val, ha="center", va="top", fontsize=12,
                  color=col, fontweight="bold", fontfamily="monospace",
                  transform=ax_t.transAxes)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL A  —  Full price history + SMA overlays + test overlay + forecast
    #             with a zoom inset in the bottom-right corner
    # ══════════════════════════════════════════════════════════════════════════
    ax_a = _style_ax(fig.add_subplot(gs[1, :]))

    train_end = test_dates[0] if len(test_dates) else prices.index[-1]
    ax_a.axvspan(prices.index[0], train_end,
                 color=TEAL, alpha=0.04, label="Train region")

    ax_a.plot(prices.index, prices.values,
              color=TEAL, linewidth=0.9, alpha=0.65, label="Historical Close")
    ax_a.plot(prices.index, prices.rolling(50).mean(),
              color=VIOLET, linewidth=0.9, linestyle="--", alpha=0.6, label="SMA-50")
    ax_a.plot(prices.index, prices.rolling(20).mean(),
              color=AMBER,  linewidth=0.9, linestyle="--", alpha=0.6, label="SMA-20")

    td = test_dates[:len(y_actual)]
    ax_a.fill_between(td, lower_band, upper_band, color=CORAL, alpha=0.12,
                      label="+/-1sigma pred band")
    ax_a.plot(td, y_actual, color=WHITE,  linewidth=1.4, alpha=0.9,
              label="Actual (test)")
    ax_a.plot(td, y_pred,   color=CORAL,  linewidth=1.3, linestyle="--",
              label="Predicted (test)")

    ax_a.plot(fut_dates, future_preds, color=AMBER, linewidth=2.2,
              marker="o", markersize=5.5, zorder=6,
              label=f"{future_days}-day Forecast")
    ax_a.fill_between(fut_dates, cone_lower, cone_upper,
                      color=AMBER, alpha=0.10, label="Forecast cone (1sigma)")
    ax_a.axvline(x=prices.index[-1], color=DIMMED, linewidth=1.1, linestyle=":")

    ax_a.set_title("A — Full Price History  |  Test Predictions  |  Future Forecast",
                   pad=7, fontsize=11)
    ax_a.set_ylabel("Price")
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_a.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate()
    ax_a.legend(ncol=5, facecolor=PANEL, edgecolor=BORDER,
                labelcolor=SLATE, fontsize=8, loc="upper left")

    # ── Forecast zoom inset ────────────────────────────────────────────────────
    ax_z = _style_ax(ax_a.inset_axes([0.69, 0.02, 0.30, 0.48]))
    zoom_n = min(60, len(prices))
    zp     = prices.iloc[-zoom_n:]
    ax_z.plot(zp.index, zp.values, color=TEAL, linewidth=1.3)
    ax_z.plot(fut_dates, future_preds, color=AMBER,
              linewidth=1.9, marker="o", markersize=4.5)
    ax_z.fill_between(fut_dates, cone_lower, cone_upper, color=AMBER, alpha=0.14)
    ax_z.axvline(prices.index[-1], color=DIMMED, linewidth=0.9, linestyle=":")
    for d, v in zip(fut_dates, future_preds):
        clr = GREEN if v >= last_price else RED
        ax_z.annotate(f"{v:.3f}", xy=(d, v), fontsize=5.5, color=clr,
                      ha="center", va="bottom", xytext=(0, 4),
                      textcoords="offset points")
    ax_z.set_title("Forecast Zoom", fontsize=8, color=WHITE, pad=3)
    ax_z.tick_params(labelsize=6)
    ax_z.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_z.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    for sp in ax_z.spines.values():
        sp.set_edgecolor(AMBER)
        sp.set_linewidth(1.5)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL B  —  Actual vs Predicted scatter  (coloured by time)
    # ══════════════════════════════════════════════════════════════════════════
    ax_b = _style_ax(fig.add_subplot(gs[2, 0]))
    cmap_seq = LinearSegmentedColormap.from_list("tp", [VIOLET, TEAL])
    sc = ax_b.scatter(y_actual, y_pred, c=np.arange(len(y_actual)),
                      cmap=cmap_seq, alpha=0.55, s=14, linewidths=0)
    lo = min(y_actual.min(), y_pred.min())
    hi = max(y_actual.max(), y_pred.max())
    ax_b.plot([lo, hi], [lo, hi], color=CORAL, linewidth=1.6,
              linestyle="--", label="Perfect fit")
    ax_b.set_xlabel("Actual price")
    ax_b.set_ylabel("Predicted price")
    ax_b.set_title("B — Actual vs Predicted  (colour = time)", fontsize=10)
    cb_b = fig.colorbar(sc, ax=ax_b, pad=0.02)
    cb_b.ax.yaxis.set_tick_params(color=DIMMED, labelsize=6)
    cb_b.set_label("Time ->", color=DIMMED, fontsize=7)
    ax_b.text(0.05, 0.93, f"R2 = {metrics['R2']:.4f}",
              transform=ax_b.transAxes, fontsize=9,
              color=TEAL, fontfamily="monospace")
    ax_b.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=SLATE, fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL C  —  Signed residuals over time  (green=over, red=under, rolling mean)
    # ══════════════════════════════════════════════════════════════════════════
    ax_c = _style_ax(fig.add_subplot(gs[2, 1]))
    td_r = test_dates[:len(residuals)]
    ax_c.fill_between(td_r, residuals, 0,
                      where=residuals >= 0, color=GREEN, alpha=0.28,
                      label="Over-predicted")
    ax_c.fill_between(td_r, residuals, 0,
                      where=residuals < 0,  color=RED,   alpha=0.28,
                      label="Under-predicted")
    ax_c.plot(td_r, residuals, color=WHITE, linewidth=0.6, alpha=0.5)
    roll_res = pd.Series(residuals).rolling(roll_win).mean()
    ax_c.plot(td_r, roll_res, color=AMBER, linewidth=1.6,
              label=f"Rolling mean ({roll_win}d)")
    ax_c.axhline(0, color=DIMMED, linewidth=1.0, linestyle="--")
    ax_c.set_ylabel("Error  (Actual - Predicted)")
    ax_c.set_title("C — Residuals Over Time", fontsize=10)
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_c.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_c.tick_params(axis="x", rotation=30)
    ax_c.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=SLATE, fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL D  —  Error distribution  (histogram + KDE + Gaussian fit)
    # ══════════════════════════════════════════════════════════════════════════
    ax_d = _style_ax(fig.add_subplot(gs[2, 2]))
    ax_d.hist(residuals, bins=38, color=TEAL, alpha=0.38, density=True,
              label="Error histogram", edgecolor=BORDER, linewidth=0.3)
    kde_x = np.linspace(residuals.min(), residuals.max(), 400)
    ax_d.plot(kde_x, gaussian_kde(residuals)(kde_x),
              color=AMBER, linewidth=2.1, label="KDE")
    mu, std = norm.fit(residuals)
    ax_d.plot(kde_x, norm.pdf(kde_x, mu, std),
              color=VIOLET, linewidth=1.5, linestyle="--", label="Normal fit")
    ax_d.axvline(0,                 color=CORAL,  linewidth=1.6,
                 linestyle="--", label="Zero error")
    ax_d.axvline(np.mean(residuals), color=WHITE, linewidth=1.1,
                 linestyle=":", label=f"Mean ({np.mean(residuals):.4f})")
    ax_d.set_xlabel("Prediction error")
    ax_d.set_ylabel("Density")
    ax_d.set_title("D — Error Distribution", fontsize=10)
    ax_d.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=SLATE, fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL E  —  Rolling MAE  +  rolling directional accuracy  (dual axis)
    # ══════════════════════════════════════════════════════════════════════════
    ax_e  = _style_ax(fig.add_subplot(gs[3, 0]))
    ax_e2 = ax_e.twinx()
    ax_e2.set_facecolor(PANEL)
    ax_e2.tick_params(colors=SLATE, labelsize=8)

    td_m = test_dates[:len(roll_mae)]
    ax_e.plot(td_m,  roll_mae, color=CORAL, linewidth=1.5, label="Rolling MAE")

    td_d = test_dates[1: len(roll_dir) + 1]
    ax_e2.plot(td_d, roll_dir, color=GREEN, linewidth=1.5,
               linestyle="--", label="Rolling Dir. Acc. %")
    ax_e2.axhline(50, color=DIMMED, linewidth=0.8, linestyle=":")

    ax_e.set_ylabel("MAE",          color=CORAL)
    ax_e2.set_ylabel("Dir. Acc. %", color=GREEN)
    ax_e.set_title("E — Rolling Performance Metrics", fontsize=10)
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_e.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_e.tick_params(axis="x", rotation=30)
    h1, l1 = ax_e.get_legend_handles_labels()
    h2, l2 = ax_e2.get_legend_handles_labels()
    ax_e.legend(h1 + h2, l1 + l2, facecolor=PANEL, edgecolor=BORDER,
                labelcolor=SLATE, fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL F  —  Simulated cumulative return: model signal vs buy-and-hold
    # ══════════════════════════════════════════════════════════════════════════
    ax_f = _style_ax(fig.add_subplot(gs[3, 1]))
    td_c = test_dates[1: len(cum_strategy) + 1]
    ax_f.plot(td_c, cum_bh       * 100, color=SLATE, linewidth=1.4,
              linestyle="--", label="Buy & Hold")
    ax_f.plot(td_c, cum_strategy * 100, color=TEAL,  linewidth=1.6,
              label="Model Signal Strategy")
    ax_f.axhline(0, color=DIMMED, linewidth=0.8, linestyle=":")
    ax_f.fill_between(td_c, cum_strategy * 100, 0,
                      where=cum_strategy >= 0, color=GREEN, alpha=0.14)
    ax_f.fill_between(td_c, cum_strategy * 100, 0,
                      where=cum_strategy < 0,  color=RED,   alpha=0.14)
    ax_f.set_ylabel("Cumulative return (%)")
    ax_f.set_title("F — Simulated Strategy vs Buy & Hold", fontsize=10)
    ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_f.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_f.tick_params(axis="x", rotation=30)
    ax_f.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=SLATE, fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL G  —  Training / validation loss  +  metrics summary table
    # ══════════════════════════════════════════════════════════════════════════
    ax_g = _style_ax(fig.add_subplot(gs[3, 2]))
    epochs_arr = np.arange(1, len(history.history["loss"]) + 1)
    ax_g.plot(epochs_arr, history.history["loss"],
              color=TEAL,  linewidth=1.6, label="Train loss")
    ax_g.plot(epochs_arr, history.history["val_loss"],
              color=CORAL, linewidth=1.6, linestyle="--", label="Val loss")
    best_ep = int(np.argmin(history.history["val_loss"])) + 1
    best_vl = min(history.history["val_loss"])
    ax_g.axvline(best_ep, color=AMBER, linewidth=1.1, linestyle=":",
                 label=f"Best epoch ({best_ep})")
    ax_g.scatter([best_ep], [best_vl], color=AMBER, s=55, zorder=6)
    ax_g.set_xlabel("Epoch")
    ax_g.set_ylabel("Huber Loss")
    ax_g.set_title("G — Training & Validation Loss  +  Metrics Summary", fontsize=10)
    ax_g.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=SLATE, fontsize=8,
                loc="upper right")

    # Metrics mini-table inset
    table_data = [
        ["MAE",       f"{metrics['MAE']:.4f}"],
        ["RMSE",      f"{metrics['RMSE']:.4f}"],
        ["MAPE",      f"{metrics['MAPE_pct']:.2f}%"],
        ["R2",        f"{metrics['R2']:.4f}"],
        ["Dir. Acc.", f"{metrics['Directional_Accuracy_pct']:.1f}%"],
        ["Best Ep.",  str(best_ep)],
    ]
    tbl = ax_g.table(
        cellText=[r[1:] for r in table_data],
        rowLabels=[r[0] for r in table_data],
        colLabels=["Value"],
        loc="lower right",
        bbox=[0.50, 0.02, 0.48, 0.58],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(BG)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(
            color=TEAL if r > 0 else WHITE,
            fontfamily="monospace")

    plt.savefig(outfile, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Report saved -> {outfile}")
    return outfile


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="LSTM currency price predictor")
    p.add_argument("--pair",   default=DEFAULTS["pair"],
                   help="Yahoo Finance ticker: EURUSD=X, GBPUSD=X, BTC-USD ...")
    p.add_argument("--period", default=DEFAULTS["period"],
                   help="History window: 1y / 2y / 5y / 10y / max")
    p.add_argument("--seq",    type=int, default=DEFAULTS["seq_len"],
                   help="Look-back window in trading days (default 60)")
    p.add_argument("--days",   type=int, default=DEFAULTS["future_days"],
                   help="Days ahead to forecast (default 7)")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch",  type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--out",    default="prediction_report.png",
                   help="Output chart filename")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 58)
    print("  Currency Price Predictor  (LSTM + 9-Panel Report)")
    print("=" * 58)

    df = fetch_data(args.pair, args.period)

    (X_tr, X_te, y_tr, y_te,
     scaler, close_scaler, scaled,
     feat_df, test_dates) = preprocess(df, args.seq, DEFAULTS["test_split"])

    model   = build_model(args.seq, len(FEAT_COLS))
    history = train_model(model, X_tr, y_tr, args.epochs, args.batch)

    y_actual, y_pred, metrics = validate(
        model, X_te, y_te, close_scaler, args.pair)

    future_preds, last_price = forecast_future(
        model, scaled, close_scaler, args.seq, args.days, len(FEAT_COLS))

    plot_results(
        feat_df, test_dates, y_actual, y_pred,
        future_preds, last_price, metrics, history,
        args.pair, args.days, args.seq, outfile=args.out)

    print("\nDone.\n")


if __name__ == "__main__":
    main()