import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import tensorflow as tf


# -----------------------------
# Helper: Ensure datetime Series is in UTC
# -----------------------------
def ensure_utc(series):
    """
    Ensure a pandas Series of datetimes is tz-aware in UTC.
    If the series is tz-naive, localize to UTC;
    if it's already tz-aware, convert to UTC.
    """
    series = pd.to_datetime(series)
    if series.dt.tz is None:
        return series.dt.tz_localize('UTC')
    else:
        return series.dt.tz_convert('UTC')


# -----------------------------
# Helper: Create Sequences for LSTM Model
# -----------------------------
def create_sequences(data, time_steps=30):
    """
    Creates sliding windows of length `time_steps` from a NumPy array.
    """
    sequences = []
    for i in range(len(data) - time_steps + 1):
        sequences.append(data[i: i + time_steps])
    return np.array(sequences)


# -----------------------------
# Helper: Load JSON Data
# -----------------------------
def load_json_data(json_path="combined_token_data.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


# -----------------------------
# -----------------------------
# Helper: Load the Saved Model
def load_model(model_path="pump_model.h5"):
    return tf.keras.models.load_model(model_path)


# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title('🐼 PANDA')
st.markdown('<p style="color: grey; font-size: 1.2rem;">Pump Analysis & Notification for Dump Avoidance</p>', unsafe_allow_html=True)

# Input: Token name (e.g., "Tap" or "Verasity")
token = st.text_input("Enter token name", value="Tap").strip().lower()

if st.button("Generate Graph with Model Predictions"):
    # Load JSON data and get token-specific data
    data = load_json_data()
    token_data = data.get(token)
    if token_data is None:
        st.error(f"Token '{token}' not found in the JSON data.")
    else:
        # ------------------------------------------
        # Extract CoinMarketCap historical price data
        # ------------------------------------------
        cmc_quotes = token_data["coinmarketcap"]["historical_data"]["quotes"]
        cmc_times = []
        prices = []
        for quote in cmc_quotes:
            ts_str = quote["timestamp"]
            dt = pd.to_datetime(ts_str.replace("Z", "+00:00"))
            cmc_times.append(dt)
            prices.append(quote["quote"]["USD"]["price"])

        if len(cmc_times) == 0:
            st.error("No price data available for this token.")
        else:
            # Create DataFrame and sort by time
            df_price = pd.DataFrame({"time": cmc_times, "price": prices}).sort_values("time")

            # For demonstration, simulate LunarCrush interactions as zeros.
            df_price['interactions'] = 0

            # -----------------------------
            # Feature Engineering
            # -----------------------------
            # Compute percentage changes as additional features
            df_price['price_pct_change'] = df_price['price'].pct_change().fillna(0)
            df_price['interactions_pct_change'] = df_price['interactions'].pct_change().fillna(0)

            # Select features for the model.
            feature_cols = ['price', 'interactions', 'price_pct_change', 'interactions_pct_change']
            data_features = df_price[feature_cols].values

            # Define the number of timesteps expected by your model
            time_steps = 30
            if len(data_features) < time_steps:
                st.error("Not enough data to create sequences for prediction.")
            else:
                # Create sequences from the features
                X_model = create_sequences(data_features, time_steps=time_steps)

                # Load the pre-trained model
                model = load_model("pump_model.h5")

                # Make predictions. The model should output a probability per sequence.
                y_pred_prob = model.predict(X_model)

                # Align predictions with timestamps.
                # We take the timestamp of the last time step of each sequence.
                pred_timestamps = df_price['time'].iloc[time_steps - 1:].reset_index(drop=True)

                # For this example, treat the predictions on the last 20% of the data as our "test set"
                n_total = len(pred_timestamps)
                test_start = int(0.8 * n_total)
                df_result_test = pd.DataFrame({
                    "time": pred_timestamps.iloc[test_start:].values,
                    "price": df_price['price'].iloc[time_steps - 1:].values[test_start:],
                    "pred_prob": y_pred_prob.flatten()[test_start:]
                })

                # ------------------------------------------
                # Filter data to only the past year
                # ------------------------------------------
                one_year_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)
                df_result_test['time'] = ensure_utc(df_result_test['time'])
                df_result_test = df_result_test[df_result_test['time'] >= one_year_ago].reset_index(drop=True)

                if df_result_test.empty:
                    st.error("No test data available in the last year.")
                else:
                    price_timestamps = df_result_test['time'].to_numpy()
                    y_tap_pred_prob = df_result_test['pred_prob'].to_numpy()

                    # ------------------------------------------
                    # Find the maximum pump probability event above threshold
                    # ------------------------------------------
                    threshold = 0  # Define your pump probability threshold
                    max_prob = -float('inf')
                    max_timestamp = None

                    st.write("Predicted pump events (timestamps with probabilities above threshold):")
                    for i, conf in enumerate(y_tap_pred_prob):
                        ts = pd.to_datetime(price_timestamps[i])
                        if conf > threshold and conf > max_prob:
                            max_prob = conf
                            max_timestamp = price_timestamps[i]
                            st.write(f"Timestamp: {ts} - Probability: {conf:.2f}")

                    if max_timestamp is None:
                        st.write("No pump events above the threshold were detected in the last year.")
                    else:
                        max_timestamp = pd.to_datetime(max_timestamp).to_pydatetime()
                        st.write(f"Max pump event: {max_timestamp} with probability {max_prob:.2f}")

                        # Define the region to highlight and a zoom window for the plot
                        region_end = max_timestamp + timedelta(days=7)
                        zoom_start = max_timestamp - timedelta(days=10)
                        zoom_end = max_timestamp + timedelta(days=20)

                        # ------------------------------------------
                        # Plot the zoomed graph
                        # ------------------------------------------
                        fig, ax1 = plt.subplots(figsize=(14, 7))
                        ax1.plot(price_timestamps, df_result_test["price"], label="Exchange Price", color="blue",
                                 linewidth=2)
                        ax1.set_xlabel("Time")
                        ax1.set_ylabel("Price (USD)", color="blue")
                        ax1.tick_params(axis="y", labelcolor="blue")

                        # Highlight the pump region: from 3 days before the max pump event until region_end
                        ax1.axvspan(max_timestamp - timedelta(days=3), region_end, color="yellow", alpha=0.3,
                                    label="Max Pump Region (10 days)")
                        ax1.set_xlim(zoom_start, zoom_end)

                        # Optionally, add a secondary axis if you have additional data (e.g., Lunar interactions)
                        ax2 = ax1.twinx()

                        # Combine legends
                        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
                        lines, labels_ll = [sum(lol, []) for lol in zip(*lines_labels)]
                        by_label = dict(zip(labels_ll, lines))
                        fig.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(0.1, 0.9))

                        plt.title(f"{token} Coin: Zoomed View with Model Pump Predictions (Last Year)")
                        plt.tight_layout()
                        st.pyplot(fig)
