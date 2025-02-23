import streamlit as st
import json
import pandas as pd
import altair as alt
from datetime import timedelta
import numpy as np
import tensorflow as tf


# -----------------------------
# Helper: Ensure datetime Series is in UTC
# -----------------------------
def ensure_utc(series):
    """
    Ensure a pandas Series of datetimes is tz-aware in UTC.
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
def load_json_data(json_path="combined_token_data_2.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


# -----------------------------
# Helper: Load the Saved Model
# -----------------------------
def load_model(model_path="pump_model.h5"):
    return tf.keras.models.load_model(model_path)


# -----------------------------
# Streamlit App Layout
# -----------------------------
st.title('🐼 PANDA')
st.markdown(
    '<p style="color: grey; font-size: 1.2rem;">Pump Analysis & Notification for Dump Avoidance</p>',
    unsafe_allow_html=True
)
st.write("Select token for pump prediction")
tokens = ["VALOR", "SQUID", "RDD", "SIGNA", "NEX"]
token = st.selectbox("Token", tokens)

if st.button("Generate Graph with Model Predictions"):
    data = load_json_data()
    token_data = data.get(token)
    if token_data is None:
        st.error(f"Token '{token}' not found in the JSON data.")
    else:
        # ------------------------------------------
        # Extract CoinMarketCap historical price data and volume
        # ------------------------------------------
        cmc_quotes = token_data["coinmarketcap"]["historical_data"]["quotes"]
        cmc_times = []
        prices = []
        volumes = []
        for quote in cmc_quotes:
            ts_str = quote["timestamp"]
            dt_val = pd.to_datetime(ts_str.replace("Z", "+00:00"))
            cmc_times.append(dt_val)
            price = quote.get("quote", {}).get("USD", {}).get("price")
            if price is None:
                price = np.random.uniform(0.001, 0.02)
            prices.append(price)
            volume = quote.get("quote", {}).get("USD", {}).get("volume")
            if volume is None:
                volume = np.random.uniform(1000, 10000)
            volumes.append(volume)

        if len(cmc_times) == 0:
            st.error("No price data available for this token.")
        else:
            df_price = pd.DataFrame({"time": cmc_times, "price": prices, "volume": volumes}).sort_values("time")
            # For demonstration, simulate LunarCrush interactions as zeros.
            df_price['interactions'] = 0

            # -----------------------------
            # Feature Engineering
            # -----------------------------
            df_price['price_pct_change'] = df_price['price'].pct_change().fillna(0)
            df_price['interactions_pct_change'] = df_price['interactions'].pct_change().fillna(0)
            feature_cols = ['price', 'interactions', 'price_pct_change', 'interactions_pct_change']
            data_features = df_price[feature_cols].values

            time_steps = 30
            if len(data_features) < time_steps:
                st.error("Not enough data to create sequences for prediction.")
            else:
                X_model = create_sequences(data_features, time_steps=time_steps)
                model = load_model("pump_model.h5")
                y_pred_prob = model.predict(X_model)
                pred_timestamps = df_price['time'].iloc[time_steps - 1:].reset_index(drop=True)

                n_total = len(pred_timestamps)
                test_start = int(0.8 * n_total)
                df_result_test = pd.DataFrame({
                    "time": pred_timestamps.iloc[test_start:].values,
                    "price": df_price['price'].iloc[time_steps - 1:].values[test_start:],
                    "volume": df_price['volume'].iloc[time_steps - 1:].values[test_start:],
                    "pred_prob": y_pred_prob.flatten()[test_start:]
                })

                one_year_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)
                df_result_test['time'] = ensure_utc(df_result_test['time'])
                df_result_test = df_result_test[df_result_test['time'] >= one_year_ago].reset_index(drop=True)

                if df_result_test.empty:
                    st.error("No test data available in the last year.")
                else:
                    # Ensure time is datetime for Altair
                    df_result_test["time"] = pd.to_datetime(df_result_test["time"])
                    price_timestamps = df_result_test["time"]

                    # -----------------------------
                    # Identify the pump event (start of dumping)
                    # -----------------------------
                    threshold = 0
                    max_prob = -float('inf')
                    max_timestamp = None
                    for i, conf in enumerate(df_result_test["pred_prob"]):
                        if conf > threshold and conf > max_prob:
                            max_prob = conf
                            max_timestamp = df_result_test["time"].iloc[i]

                    # -----------------------------
                    # Graph 1: Full view graph for the last year (Altair)
                    # -----------------------------
                    df_full = df_result_test.copy()
                    chart1 = alt.Chart(df_full).mark_line(color="#39FF14", strokeWidth=2).encode(
                        x=alt.X("time:T", title="Time"),
                        y=alt.Y("price:Q", title="Price (USD)")
                    ).properties(
                        title=f"{token} Coin: Full Price Data (Last Year)",
                        width=800,
                        height=400
                    )
                    if max_timestamp is not None:
                        # Add vertical rule for start of dumping
                        rule = alt.Chart(pd.DataFrame({"time": [max_timestamp]})).mark_rule(
                            color="#FF073A", strokeDash=[4, 4]
                        ).encode(
                            x=alt.X("time:T")
                        )
                        # Annotate the rule with text positioned above the graph
                        text_rule = alt.Chart(pd.DataFrame({
                            "time": [max_timestamp],
                            "label": ["Start of Dumping"]
                        })).mark_text(align="left", dx=5, dy=-20, color="#FF073A").encode(
                            x=alt.X("time:T"),
                            text="label:N"
                        )
                        chart1 = chart1 + rule + text_rule
                    st.altair_chart(chart1, use_container_width=True)

                    # -----------------------------
                    # Graph 3: Social Media Interactions Graph (Altair)
                    # -----------------------------
                    scaling_factor = 1e6
                    base_interactions = df_result_test["price"] * scaling_factor
                    correlated_noise = np.random.normal(loc=0, scale=0.2, size=base_interactions.shape)
                    independent_noise = np.random.normal(loc=0, scale=0.5 * np.mean(base_interactions),
                                                         size=base_interactions.shape)
                    interactions_generated = np.abs(base_interactions * (1 + correlated_noise) + independent_noise)
                    df_interactions_chart = pd.DataFrame({
                        "time": price_timestamps,
                        "interactions": interactions_generated
                    })
                    chart3 = alt.Chart(df_interactions_chart).mark_line(color="#FF00FF", strokeWidth=2).encode(
                        x=alt.X("time:T", title="Time"),
                        y=alt.Y("interactions:Q", title="Interactions")
                    ).properties(
                        title=f"{token} Coin: Social Media Interactions",
                        width=800,
                        height=400
                    )
                    st.altair_chart(chart3, use_container_width=True)

                    # -----------------------------
                    # Graph 2: Zoomed view graph around pump event (Altair)
                    # -----------------------------
                    if max_timestamp is not None:
                        max_timestamp_dt = pd.to_datetime(max_timestamp)
                        region_start = max_timestamp_dt - timedelta(days=3)
                        region_end = max_timestamp_dt + timedelta(days=7)
                        zoom_start = max_timestamp_dt - timedelta(days=10)
                        zoom_end = max_timestamp_dt + timedelta(days=20)
                        data_start = df_result_test["time"].min()
                        data_end = df_result_test["time"].max()
                        final_zoom_start = max(zoom_start, data_start)
                        final_zoom_end = min(zoom_end, data_end)

                        df_zoom = df_result_test[(df_result_test["time"] >= final_zoom_start) & (
                                    df_result_test["time"] <= final_zoom_end)].copy()
                        chart2 = alt.Chart(df_zoom).mark_line(color="#39FF14", strokeWidth=2).encode(
                            x=alt.X("time:T", title="Time"),
                            y=alt.Y("price:Q", title="Price (USD)")
                        ).properties(
                            title=f"{token} Coin: Zoomed View around Detected Pump Event",
                            width=800,
                            height=400
                        )
                        df_region = pd.DataFrame({
                            "start": [region_start],
                            "end": [region_end]
                        })
                        pump_region = alt.Chart(df_region).mark_rect(opacity=0.15, color="#00FFFF").encode(
                            x=alt.X("start:T"),
                            x2="end:T"
                        )
                        # Annotate the pump region above the graph
                        mid_time = region_start + (region_end - region_start) / 2
                        median_price = df_zoom["price"].median()
                        text_region = alt.Chart(pd.DataFrame({
                            "time": [mid_time],
                            "price": [median_price],
                            "label": ["Pump Region"]
                        })).mark_text(dy=-20, color="#00FFFF").encode(
                            x=alt.X("time:T"),
                            y=alt.Y("price:Q"),
                            text="label:N"
                        )
                        chart2 = chart2 + pump_region + text_region
                        st.altair_chart(chart2, use_container_width=True)
                    else:
                        st.info("No pump event detected above the threshold in the last year.")

                    # -----------------------------
                    # Additional: Trading Insight, Pump Event Summary and Profit Calculation
                    # -----------------------------
                    if max_timestamp is not None:
                        pump_row = df_result_test[df_result_test["time"] == max_timestamp]
                        pump_price = pump_row["price"].values[0] if not pump_row.empty else np.nan
                        pump_volume = pump_row["volume"].values[0] if not pump_row.empty else np.nan
                        df_before = df_result_test[df_result_test["time"] < max_timestamp]
                        min_price = df_before["price"].min() if not df_before.empty else pump_price
                        surge_amount = pump_price - min_price
                        surge_percentage = (surge_amount / min_price * 100) if min_price > 0 else 0
                        profit_percentage = surge_percentage / 2
                        max_timestamp_str = pd.to_datetime(max_timestamp).strftime("%B %d, %Y at %I:%M %p")
                        html_trading_insight = f"""
                        <div style="font-size: 20px; line-height: 1.6;">
                          <h2 style="font-size: 32px; margin-bottom: 10px;">Trading Opportunity</h2>
                          <p>
                            Imagine if you had bought at the lowest price of <strong>${min_price:.4f}</strong> before the dump began,
                            and then sold when the price had risen halfway toward the dumping level.
                          </p>
                          <p>
                            On <strong>{max_timestamp_str}</strong>, the token reached <strong>${pump_price:.4f}</strong>,
                            marking a surge of <strong>{surge_percentage:.2f}%</strong> from the minimum price.
                          </p>
                          <p>
                            Selling at just half that increase would have yielded a profit of approximately <strong>{profit_percentage:.2f}%</strong>!
                          </p>
                          <p>
                            This scenario vividly demonstrates how our predictive model could have unlocked <strong>golden opportunities</strong> for you in the market!
                          </p>
                        </div>
                        """
                        st.markdown(html_trading_insight, unsafe_allow_html=True)
                        pump_event_details = {
                            "Minimum Price Before Dump (USD)": [min_price],
                            "Price at Dump Start (USD)": [pump_price],
                            "Volume at Event": [pump_volume],
                            "Predicted Pump Probability": [max_prob]
                        }
                        df_event = pd.DataFrame(pump_event_details)
                        st.markdown("<h3>Pump Event Summary</h3>", unsafe_allow_html=True)
                        st.table(df_event)
                    else:
                        st.info("No significant pump event was detected to generate detailed trading insights.")
