# Currency price predictor

## Pipeline

1. Data — pulls live historical prices from Yahoo Finance via yfinance. Works for any forex pair (EURUSD=X, GBPUSD=X) or crypto (BTC-USD, ETH-USD).
2. Preprocessing — normalises prices with MinMaxScaler, then slides a configurable look-back window (default 60 trading days) to build supervised sequences.
3. Model — a stacked LSTM with 3 recurrent layers + Dropout, compiled with the Huber loss (more robust to price spikes than MSE). Early stopping and learning-rate reduction prevent overfitting.
4. Validation — held-out test set is evaluated on five metrics:
- MAE / RMSE — absolute price error
- MAPE — percentage error, scale-independent
- R² — how much variance is explained
- Directional accuracy — did the model correctly predict up or down each day? (most relevant for trading signals)
5. Forecast — autoregressively predicts N days beyond the last known price, showing each day's value and % change.


## Install and Run

`bash pip install yfinance numpy pandas scikit-learn tensorflow matplotlib`

`# EUR/USD, 7-day forecast (default)`
`python currency_predictor.py`

`# GBP/USD, 14-day forecast`
`python currency_predictor.py --pair GBPUSD=X --days 14`

`# Bitcoin, 10-day forecast, 10 years of history`
`python currency_predictor.py --pair BTC-USD --period 10y --days 10`