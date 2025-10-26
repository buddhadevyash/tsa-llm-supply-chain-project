import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Time series forecasting imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats

# LLM imports
import os
from dotenv import load_dotenv
import groq

# Page configuration
st.set_page_config(
    page_title="Advanced Supply Chain Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment variables
load_dotenv()

# Enhanced Custom CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(255,255,255,0) 100%);
        padding: 1rem;
        border-radius: 5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    .insight-box h3 {
        color: #2c3e50;
        margin-top: 0;
        font-weight: 700;
    }
    
    .ai-response {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e0e6ed;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
    }
    
    .ai-response h4 {
        color: #667eea;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .forecast-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(9, 132, 227, 0.3);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e0e6ed;
        margin: 0.5rem 0;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stat-card h4 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stExpander"] {
        background: white;
        border-radius: 10px;
        border: 2px solid #e0e6ed;
        margin: 1rem 0;
    }
    
    .model-comparison {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class SupplyChainAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Load and preprocess the supply chain data"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.clean_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")

    def clean_data(self):
        """Clean and preprocess the data"""
        try:
            # Rename columns for consistency
            column_mapping = {
                "ProcuredDate": "Procured_Date",
                "OrderDate": "Order_Date",
                "ShipDate": "Ship_Date",
                "DeliveryDate": "Delivery_Date",
                "Sales Channel": "Sales_Channel",
                "Order Quantity": "Order_Quantity",
                "Discount Applied": "Discount_Applied",
                "Unit Cost": "Unit_Cost",
                "Unit Price": "Unit_Price",
            }
            self.df.rename(columns=column_mapping, inplace=True)

            # Convert date columns
            date_columns = ["Procured_Date", "Order_Date", "Ship_Date", "Delivery_Date"]
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

            # Convert numeric columns
            numeric_columns = [
                "Order_Quantity",
                "Unit_Cost",
                "Unit_Price",
                "Discount_Applied",
            ]
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            # Calculate derived metrics
            if all(col in self.df.columns for col in ["Order_Quantity", "Unit_Price"]):
                self.df["Total_Revenue"] = (
                    self.df["Order_Quantity"] * self.df["Unit_Price"]
                )

            if all(col in self.df.columns for col in ["Order_Date", "Delivery_Date"]):
                self.df["Lead_Time"] = (
                    self.df["Delivery_Date"] - self.df["Order_Date"]
                ).dt.days

            if all(col in self.df.columns for col in ["Order_Date", "Ship_Date"]):
                self.df["Processing_Time"] = (
                    self.df["Ship_Date"] - self.df["Order_Date"]
                ).dt.days

            if all(col in self.df.columns for col in ["Ship_Date", "Delivery_Date"]):
                self.df["Shipping_Time"] = (
                    self.df["Delivery_Date"] - self.df["Ship_Date"]
                ).dt.days

            # Remove invalid rows
            self.df = self.df[self.df["Order_Date"].notna()]

            # Add time-based features
            self.df["Year"] = self.df["Order_Date"].dt.year
            self.df["Month"] = self.df["Order_Date"].dt.month
            self.df["Quarter"] = self.df["Order_Date"].dt.quarter
            self.df["Week"] = self.df["Order_Date"].dt.isocalendar().week
            self.df["DayOfWeek"] = self.df["Order_Date"].dt.dayofweek

        except Exception as e:
            st.error(f"Error cleaning data: {e}")

    def get_time_series_data(self, date_range=None, freq="M"):
        """Get aggregated time series data"""
        try:
            df_filtered = self.df.copy()

            if date_range:
                df_filtered = df_filtered[
                    (df_filtered["Order_Date"] >= date_range[0])
                    & (df_filtered["Order_Date"] <= date_range[1])
                ]

            # Aggregate by frequency
            df_filtered["Period"] = df_filtered["Order_Date"].dt.to_period(freq)

            ts_data = (
                df_filtered.groupby("Period")
                .agg(
                    {
                        "Total_Revenue": "sum",
                        "Order_Quantity": "sum",
                        "OrderNumber": "count",
                        "Unit_Price": "mean",
                        "Lead_Time": "mean",
                        "Discount_Applied": "mean",
                    }
                )
                .reset_index()
            )

            ts_data["Period"] = ts_data["Period"].dt.to_timestamp()

            return ts_data

        except Exception as e:
            st.error(f"Error in time series aggregation: {e}")
            return pd.DataFrame()

    def perform_stationarity_test(self, series):
        """Perform Augmented Dickey-Fuller test for stationarity"""
        try:
            result = adfuller(series.dropna())
            return {
                "ADF_Statistic": result[0],
                "p_value": result[1],
                "Critical_Values": result[4],
                "is_stationary": result[1] < 0.05,
            }
        except:
            return None

    def arima_forecast(self, data, column="Total_Revenue", order=(2, 1, 2), periods=6):
        """ARIMA forecasting with confidence intervals"""
        try:
            values = data[column].values

            # Fit ARIMA model
            model = ARIMA(values, order=order)
            model_fit = model.fit()

            # Generate forecast with confidence intervals
            forecast_result = model_fit.forecast(steps=periods)
            forecast_df = model_fit.get_forecast(steps=periods).summary_frame()

            return {
                "forecast": forecast_result.tolist(),
                "lower_bound": forecast_df["mean_ci_lower"].tolist(),
                "upper_bound": forecast_df["mean_ci_upper"].tolist(),
                "model_summary": model_fit.summary(),
                "aic": model_fit.aic,
                "bic": model_fit.bic,
            }

        except Exception as e:
            st.warning(f"ARIMA forecast failed: {e}")
            return None

    def sarima_forecast(
        self,
        data,
        column="Total_Revenue",
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        periods=6,
    ):
        """SARIMA forecasting for seasonal data"""
        try:
            values = data[column].values

            # Fit SARIMA model
            model = SARIMAX(values, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)

            # Generate forecast
            forecast_result = model_fit.forecast(steps=periods)
            forecast_df = model_fit.get_forecast(steps=periods).summary_frame()

            return {
                "forecast": forecast_result.tolist(),
                "lower_bound": forecast_df["mean_ci_lower"].tolist(),
                "upper_bound": forecast_df["mean_ci_upper"].tolist(),
                "aic": model_fit.aic,
                "bic": model_fit.bic,
            }

        except Exception as e:
            st.warning(f"SARIMA forecast failed: {e}")
            return None

    def holt_winters_forecast(self, data, column="Total_Revenue", periods=6):
        """Holt-Winters exponential smoothing forecast"""
        try:
            values = data[column].values

            if len(values) < 8:
                return None

            # Determine seasonal periods
            seasonal_periods = min(4, len(values) // 2)

            # Fit Holt-Winters model
            model = ExponentialSmoothing(
                values, trend="add", seasonal="add", seasonal_periods=seasonal_periods
            )
            model_fit = model.fit()

            # Generate forecast
            forecast = model_fit.forecast(periods)

            return {
                "forecast": forecast.tolist(),
                "aic": model_fit.aic if hasattr(model_fit, "aic") else None,
            }

        except Exception as e:
            st.warning(f"Holt-Winters forecast failed: {e}")
            return None

    def calculate_metrics(self, actual, predicted):
        """Calculate forecast accuracy metrics"""
        try:
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)

            # Avoid division by zero
            actual_nonzero = actual[actual != 0]
            predicted_nonzero = predicted[actual != 0]

            if len(actual_nonzero) > 0:
                mape = (
                    np.mean(
                        np.abs((actual_nonzero - predicted_nonzero) / actual_nonzero)
                    )
                    * 100
                )
            else:
                mape = 0

            r2 = r2_score(actual, predicted)

            return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}
        except:
            return {}


class LLMAssistant:
    def __init__(self):
        self.client = None
        self.setup_llm()

    def setup_llm(self):
        """Initialize Groq client"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.client = groq.Groq(api_key=api_key)
            else:
                st.warning("⚠️ GROQ_API_KEY not found. Please add it to your .env file.")
        except Exception as e:
            st.error(f"Error setting up LLM: {e}")

    def generate_chart_summary(self, chart_type, data_summary):
        """Generate AI summary for charts"""
        if not self.client:
            return "AI summary unavailable. Please configure your API key."

        try:
            prompt = f"""
            You are a supply chain analytics expert. Generate a concise, insightful summary for the following {chart_type}.
            
            Data Summary:
            {data_summary}
            
            Provide:
            1. Key findings (2-3 bullet points)
            2. Notable trends or patterns
            3. Actionable recommendation
            
            Keep it professional and data-driven. Format with markdown.
            """

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=500,
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {e}"

    def get_detailed_analysis(self, prompt, context=""):
        """Get detailed analysis from LLM"""
        if not self.client:
            return "LLM service not available. Please check your API key."

        try:
            full_prompt = f"""
            You are an expert supply chain data scientist with deep knowledge of time series analysis, forecasting, and operations optimization.
            
            Dataset Context:
            {context}
            
            User Question: {prompt}
            
            Provide a comprehensive, well-structured analysis with:
            - Clear headings and sections
            - Specific numerical insights where applicable
            - Statistical interpretations
            - Actionable recommendations
            - Risk factors to consider
            
            Format your response in markdown with proper structure.
            """

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.4,
                max_tokens=2048,
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting response: {e}"


def show_dashboard(analyzer, llm_assistant):
    """Display the main analytics dashboard"""

    st.markdown(
        '<div class="main-header">📊 Advanced Supply Chain Analytics Platform</div>',
        unsafe_allow_html=True,
    )

    if analyzer.df is None or analyzer.df.empty:
        st.error("❌ No data loaded. Please check the data file path.")
        return

    # Sidebar filters
    st.sidebar.markdown(
        '<div class="sidebar-info"><h3>🎛️ Dashboard Controls</h3></div>',
        unsafe_allow_html=True,
    )

    # Date range filter
    min_date = analyzer.df["Order_Date"].min()
    max_date = analyzer.df["Order_Date"].max()

    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Frequency selection
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
    frequency = st.sidebar.selectbox(
        "📊 Time Aggregation", list(freq_map.keys()), index=2
    )

    # Channel filter
    if "Sales_Channel" in analyzer.df.columns:
        channels = ["All"] + list(analyzer.df["Sales_Channel"].unique())
        selected_channel = st.sidebar.selectbox("🛍️ Sales Channel", channels)
    else:
        selected_channel = "All"

    # Filter data
    filtered_df = analyzer.df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["Order_Date"] >= pd.Timestamp(date_range[0]))
            & (filtered_df["Order_Date"] <= pd.Timestamp(date_range[1]))
        ]

    if selected_channel != "All" and "Sales_Channel" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Sales_Channel"] == selected_channel]

    # Update analyzer's df temporarily
    original_df = analyzer.df
    analyzer.df = filtered_df

    # Key Metrics
    st.markdown(
        '<div class="section-header">📈 Key Performance Indicators</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_revenue = (
            filtered_df["Total_Revenue"].sum()
            if "Total_Revenue" in filtered_df.columns
            else 0
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>💰 Total Revenue</h3>
            <h2>${total_revenue:,.0f}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        total_orders = len(filtered_df)
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>📦 Total Orders</h3>
            <h2>{total_orders:,}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        avg_lead_time = (
            filtered_df["Lead_Time"].mean() if "Lead_Time" in filtered_df.columns else 0
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>⏱️ Avg Lead Time</h3>
            <h2>{avg_lead_time:.1f} days</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        avg_order_value = (
            filtered_df["Total_Revenue"].mean()
            if "Total_Revenue" in filtered_df.columns
            else 0
        )
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>💵 Avg Order Value</h3>
            <h2>${avg_order_value:,.0f}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Get time series data
    ts_data = analyzer.get_time_series_data(
        date_range if len(date_range) == 2 else None, freq_map[frequency]
    )

    if ts_data.empty:
        st.warning("No data available for the selected filters.")
        analyzer.df = original_df
        return

    # Time Series Analysis
    st.markdown(
        '<div class="section-header">📊 Time Series Analysis & Forecasting</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # Main time series plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=ts_data["Period"],
                y=ts_data["Total_Revenue"],
                mode="lines+markers",
                name="Revenue",
                line=dict(color="#667eea", width=3),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="📈 Revenue Over Time",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            template="plotly_white",
            height=400,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Statistics box
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Statistical Summary")

        if len(ts_data) > 1:
            revenue_series = ts_data["Total_Revenue"]
            st.metric("Mean Revenue", f"${revenue_series.mean():,.0f}")
            st.metric("Std Deviation", f"${revenue_series.std():,.0f}")
            st.metric(
                "Trend",
                f"{((revenue_series.iloc[-1] / revenue_series.iloc[0]) - 1) * 100:.1f}%",
            )

            # Stationarity test
            stationarity = analyzer.perform_stationarity_test(revenue_series)
            if stationarity:
                is_stationary = "✅ Yes" if stationarity["is_stationary"] else "❌ No"
                st.metric("Stationary", is_stationary)

        st.markdown("</div>", unsafe_allow_html=True)

    # AI-Generated Summary for Time Series
    with st.expander("🤖 AI-Generated Insights", expanded=True):
        if st.button("Generate AI Summary for Time Series", key="ts_summary"):
            data_summary = f"""
            Time Period: {ts_data['Period'].min()} to {ts_data['Period'].max()}
            Total Revenue: ${ts_data['Total_Revenue'].sum():,.0f}
            Average Revenue: ${ts_data['Total_Revenue'].mean():,.0f}
            Number of Periods: {len(ts_data)}
            Growth Rate: {((ts_data['Total_Revenue'].iloc[-1] / ts_data['Total_Revenue'].iloc[0]) - 1) * 100:.1f}%
            Volatility (CV): {(ts_data['Total_Revenue'].std() / ts_data['Total_Revenue'].mean()) * 100:.1f}%
            """

            summary = llm_assistant.generate_chart_summary(
                "Time Series Revenue Chart", data_summary
            )
            st.markdown(
                f'<div class="ai-response">{summary}</div>', unsafe_allow_html=True
            )

    # Advanced Forecasting Section
    st.markdown(
        '<div class="section-header">🔮 Advanced Forecasting Models</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        forecast_periods = st.slider("Forecast Periods", 3, 12, 6)
    with col2:
        arima_order = st.selectbox(
            "ARIMA Order (p,d,q)", [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1)], index=1
        )
    with col3:
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)

    # Generate forecasts
    with st.spinner("🔄 Running forecasting models..."):
        arima_result = analyzer.arima_forecast(
            ts_data, "Total_Revenue", arima_order, forecast_periods
        )
        hw_result = analyzer.holt_winters_forecast(
            ts_data, "Total_Revenue", forecast_periods
        )

        # Create forecast visualization
        fig_forecast = go.Figure()

        # Historical data
        fig_forecast.add_trace(
            go.Scatter(
                x=ts_data["Period"],
                y=ts_data["Total_Revenue"],
                mode="lines+markers",
                name="Historical",
                line=dict(color="#2c3e50", width=3),
            )
        )

        # Generate future periods
        last_period = ts_data["Period"].iloc[-1]
        if freq_map[frequency] == "M":
            future_periods = pd.date_range(
                start=last_period, periods=forecast_periods + 1, freq="MS"
            )[1:]
        elif freq_map[frequency] == "W":
            future_periods = pd.date_range(
                start=last_period, periods=forecast_periods + 1, freq="W"
            )[1:]
        else:
            future_periods = pd.date_range(
                start=last_period, periods=forecast_periods + 1, freq="D"
            )[1:]

        # ARIMA forecast
        if arima_result:
            fig_forecast.add_trace(
                go.Scatter(
                    x=future_periods,
                    y=arima_result["forecast"],
                    mode="lines+markers",
                    name="ARIMA",
                    line=dict(color="#e74c3c", width=2, dash="dash"),
                )
            )

            if show_confidence and "lower_bound" in arima_result:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=future_periods,
                        y=arima_result["upper_bound"],
                        mode="lines",
                        name="Upper Bound",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=future_periods,
                        y=arima_result["lower_bound"],
                        mode="lines",
                        name="95% CI",
                        fill="tonexty",
                        fillcolor="rgba(231, 76, 60, 0.2)",
                        line=dict(width=0),
                    )
                )

        # Holt-Winters forecast
        if hw_result:
            fig_forecast.add_trace(
                go.Scatter(
                    x=future_periods,
                    y=hw_result["forecast"],
                    mode="lines+markers",
                    name="Holt-Winters",
                    line=dict(color="#27ae60", width=2, dash="dot"),
                )
            )

        fig_forecast.update_layout(
            title="🔮 Multi-Model Revenue Forecasting",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

    # Model Comparison
    col1, col2 = st.columns(2)

    with col1:
        if arima_result:
            st.markdown(
                f"""
            <div class="forecast-card">
                <h4>📊 ARIMA Model</h4>
                <p><strong>Order:</strong> {arima_order}</p>
                <p><strong>AIC:</strong> {arima_result.get('aic', 'N/A'):.2f}</p>
                <p><strong>BIC:</strong> {arima_result.get('bic', 'N/A'):.2f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        if hw_result:
            st.markdown(
                f"""
            <div class="forecast-card">
                <h4>📈 Holt-Winters Model</h4>
                <p><strong>Seasonal:</strong> Additive</p>
                <p><strong>AIC:</strong> {hw_result.get('aic', 'N/A'):.2f if hw_result.get('aic') else 'N/A'}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Advanced Time Series Decomposition
    st.markdown(
        '<div class="section-header">🔍 Time Series Decomposition</div>',
        unsafe_allow_html=True,
    )

    if len(ts_data) >= 24:  # Minimum data points for decomposition
        try:
            # Prepare data for decomposition
            ts_series = ts_data.set_index("Period")["Total_Revenue"]

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                ts_series, model="additive", period=min(12, len(ts_data) // 2)
            )

            # Create decomposition plot
            fig_decomp = make_subplots(
                rows=4,
                cols=1,
                subplot_titles=[
                    "Original Series",
                    "Trend Component",
                    "Seasonal Component",
                    "Residual Component",
                ],
                vertical_spacing=0.08,
            )

            # Original series
            fig_decomp.add_trace(
                go.Scatter(
                    x=ts_series.index,
                    y=ts_series.values,
                    name="Original",
                    line=dict(color="#667eea"),
                ),
                row=1,
                col=1,
            )

            # Trend
            fig_decomp.add_trace(
                go.Scatter(
                    x=decomposition.trend.index,
                    y=decomposition.trend.values,
                    name="Trend",
                    line=dict(color="#e74c3c"),
                ),
                row=2,
                col=1,
            )

            # Seasonal
            fig_decomp.add_trace(
                go.Scatter(
                    x=decomposition.seasonal.index,
                    y=decomposition.seasonal.values,
                    name="Seasonal",
                    line=dict(color="#27ae60"),
                ),
                row=3,
                col=1,
            )

            # Residual
            fig_decomp.add_trace(
                go.Scatter(
                    x=decomposition.resid.index,
                    y=decomposition.resid.values,
                    name="Residual",
                    line=dict(color="#f39c12"),
                    mode="markers",
                ),
                row=4,
                col=1,
            )

            fig_decomp.update_layout(
                height=800, showlegend=False, template="plotly_white"
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

            # ACF and PACF plots
            st.markdown(
                '<div class="section-header">📊 Autocorrelation Analysis</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                # ACF Plot
                fig_acf = go.Figure()
                acf_vals, confint = acf(
                    ts_series.dropna(), nlags=min(20, len(ts_series) // 4), alpha=0.05
                )

                fig_acf.add_trace(
                    go.Bar(
                        x=list(range(len(acf_vals))),
                        y=acf_vals,
                        name="ACF",
                        marker_color="#3498db",
                    )
                )

                # Add confidence intervals
                fig_acf.add_trace(
                    go.Scatter(
                        x=list(range(len(acf_vals))) + list(range(len(acf_vals)))[::-1],
                        y=list(confint[:, 0] - acf_vals)
                        + list(confint[:, 1] - acf_vals)[::-1],
                        fill="toself",
                        fillcolor="rgba(52, 152, 219, 0.3)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="95% Confidence",
                    )
                )

                fig_acf.update_layout(
                    title="Autocorrelation Function (ACF)",
                    xaxis_title="Lag",
                    yaxis_title="ACF",
                    template="plotly_white",
                )
                st.plotly_chart(fig_acf, use_container_width=True)

            with col2:
                # PACF Plot
                fig_pacf = go.Figure()
                pacf_vals, confint = pacf(
                    ts_series.dropna(), nlags=min(20, len(ts_series) // 4), alpha=0.05
                )

                fig_pacf.add_trace(
                    go.Bar(
                        x=list(range(len(pacf_vals))),
                        y=pacf_vals,
                        name="PACF",
                        marker_color="#9b59b6",
                    )
                )

                # Add confidence intervals
                fig_pacf.add_trace(
                    go.Scatter(
                        x=list(range(len(pacf_vals)))
                        + list(range(len(pacf_vals)))[::-1],
                        y=list(confint[:, 0] - pacf_vals)
                        + list(confint[:, 1] - pacf_vals)[::-1],
                        fill="toself",
                        fillcolor="rgba(155, 89, 182, 0.3)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="95% Confidence",
                    )
                )

                fig_pacf.update_layout(
                    title="Partial Autocorrelation Function (PACF)",
                    xaxis_title="Lag",
                    yaxis_title="PACF",
                    template="plotly_white",
                )
                st.plotly_chart(fig_pacf, use_container_width=True)

        except Exception as e:
            st.warning(f"Decomposition not possible: {e}")

    # Supply Chain Performance Metrics
    st.markdown(
        '<div class="section-header">⛓️ Supply Chain Performance</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if "Lead_Time" in filtered_df.columns:
            lead_time_stats = filtered_df["Lead_Time"].describe()
            st.plotly_chart(
                create_distribution_plot(
                    filtered_df, "Lead_Time", "Lead Time Distribution", "#e67e22"
                ),
                use_container_width=True,
            )

    with col2:
        if "Processing_Time" in filtered_df.columns:
            st.plotly_chart(
                create_distribution_plot(
                    filtered_df,
                    "Processing_Time",
                    "Processing Time Distribution",
                    "#3498db",
                ),
                use_container_width=True,
            )

    with col3:
        if "Shipping_Time" in filtered_df.columns:
            st.plotly_chart(
                create_distribution_plot(
                    filtered_df,
                    "Shipping_Time",
                    "Shipping Time Distribution",
                    "#9b59b6",
                ),
                use_container_width=True,
            )

    # Advanced Analytics Section
    st.markdown(
        '<div class="section-header">🧠 Advanced Analytics</div>',
        unsafe_allow_html=True,
    )

    # Model Selection for Detailed Analysis
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
        ["📈 SARIMA Analysis", "🔍 Anomaly Detection", "📊 Cross-Correlation"]
    )

    with analysis_tab1:
        st.subheader("SARIMA Model Configuration")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sarima_p = st.slider("AR Order (p)", 0, 3, 1)
        with col2:
            sarima_d = st.slider("Difference Order (d)", 0, 2, 1)
        with col3:
            sarima_q = st.slider("MA Order (q)", 0, 3, 1)
        with col4:
            sarima_s = st.slider("Seasonal Period", 4, 12, 12)

        if st.button("Run SARIMA Analysis", key="sarima_analysis"):
            with st.spinner("Training SARIMA model..."):
                sarima_result = analyzer.sarima_forecast(
                    ts_data,
                    "Total_Revenue",
                    order=(sarima_p, sarima_d, sarima_q),
                    seasonal_order=(1, 1, 1, sarima_s),
                    periods=forecast_periods,
                )

                if sarima_result:
                    # Display SARIMA results
                    st.markdown(
                        f"""
                    <div class="model-comparison">
                        <h4>🎯 SARIMA Model Results</h4>
                        <p><strong>Order:</strong> ({sarima_p},{sarima_d},{sarima_q}) × (1,1,1,{sarima_s})</p>
                        <p><strong>AIC:</strong> {sarima_result.get('aic', 'N/A'):.2f}</p>
                        <p><strong>BIC:</strong> {sarima_result.get('bic', 'N/A'):.2f}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # SARIMA forecast plot
                    fig_sarima = create_sarima_forecast_plot(
                        ts_data, sarima_result, future_periods
                    )
                    st.plotly_chart(fig_sarima, use_container_width=True)

    with analysis_tab2:
        st.subheader("Anomaly Detection in Time Series")

        if st.button("Detect Anomalies"):
            anomalies = detect_anomalies(ts_data, "Total_Revenue")
            fig_anomalies = create_anomaly_plot(ts_data, anomalies)
            st.plotly_chart(fig_anomalies, use_container_width=True)

            # Anomaly summary
            if len(anomalies) > 0:
                st.warning(
                    f"🚨 {len(anomalies)} anomalies detected in the time series!"
                )

                # AI analysis of anomalies
                with st.expander("🤖 AI Anomaly Analysis"):
                    anomaly_summary = f"""
                    Detected {len(anomalies)} anomalies in revenue data.
                    Anomaly periods: {', '.join([str(ts_data.iloc[i]['Period'].strftime('%Y-%m')) for i in anomalies])}
                    Average revenue during anomalies: ${ts_data.iloc[anomalies]['Total_Revenue'].mean():,.0f}
                    Impact on overall revenue: {(ts_data.iloc[anomalies]['Total_Revenue'].sum() / ts_data['Total_Revenue'].sum()) * 100:.1f}%
                    """

                    analysis = llm_assistant.generate_chart_summary(
                        "Anomaly Detection Results", anomaly_summary
                    )
                    st.markdown(
                        f'<div class="ai-response">{analysis}</div>',
                        unsafe_allow_html=True,
                    )

    with analysis_tab3:
        st.subheader("Cross-Correlation Analysis")

        if all(
            col in ts_data.columns
            for col in ["Total_Revenue", "Order_Quantity", "Discount_Applied"]
        ):
            fig_corr = create_correlation_heatmap(
                ts_data[
                    ["Total_Revenue", "Order_Quantity", "Discount_Applied", "Lead_Time"]
                ].corr()
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Lag correlation analysis
            st.subheader("Lag Correlation Analysis")
            max_lag = st.slider("Maximum Lag", 1, 12, 6)

            if (
                "Total_Revenue" in ts_data.columns
                and "Order_Quantity" in ts_data.columns
            ):
                fig_lag = create_lag_correlation_plot(
                    ts_data, "Total_Revenue", "Order_Quantity", max_lag
                )
                st.plotly_chart(fig_lag, use_container_width=True)

    # Interactive AI Assistant
    st.markdown(
        '<div class="section-header">🤖 AI Supply Chain Assistant</div>',
        unsafe_allow_html=True,
    )

    user_question = st.text_area(
        "💬 Ask any question about your supply chain data:",
        placeholder="E.g., What are the key factors affecting our lead times? How can we optimize inventory based on seasonal patterns?",
        height=100,
    )

    if st.button("Get AI Analysis", key="ai_analysis"):
        if user_question:
            with st.spinner("🤔 Analyzing your data..."):
                context = f"""
                Data Overview:
                - Time Range: {ts_data['Period'].min()} to {ts_data['Period'].max()}
                - Total Revenue: ${ts_data['Total_Revenue'].sum():,.0f}
                - Total Orders: {len(filtered_df):,}
                - Average Lead Time: {filtered_df['Lead_Time'].mean() if 'Lead_Time' in filtered_df.columns else 'N/A'} days
                - Key Metrics: Revenue growth, order patterns, seasonal trends
                
                Forecasting Results:
                - ARIMA Performance: {arima_result.get('aic', 'N/A') if arima_result else 'N/A'}
                - Seasonal Patterns: {'Detected' if len(ts_data) >= 24 else 'Insufficient data'}
                - Data Stationarity: {stationarity.get('is_stationary') if stationarity else 'Unknown'}
                """

                response = llm_assistant.get_detailed_analysis(user_question, context)
                st.markdown(
                    f'<div class="ai-response">{response}</div>', unsafe_allow_html=True
                )
        else:
            st.warning("Please enter a question for the AI assistant.")

    # Data Export and Reporting
    st.markdown(
        '<div class="section-header">📤 Reports & Export</div>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Generate Comprehensive Report"):
            generate_comprehensive_report(analyzer, ts_data, arima_result, hw_result)

    with col2:
        # Data download
        csv = ts_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Time Series Data",
            data=csv,
            file_name=f"supply_chain_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    # Restore original data
    analyzer.df = original_df


def create_distribution_plot(df, column, title, color):
    """Create distribution plot for supply chain metrics"""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df[column].dropna(),
            nbinsx=20,
            name=title,
            marker_color=color,
            opacity=0.7,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title="Frequency",
        template="plotly_white",
        height=300,
    )

    return fig


def create_sarima_forecast_plot(historical_data, sarima_result, future_periods):
    """Create SARIMA forecast visualization"""
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data["Period"],
            y=historical_data["Total_Revenue"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="#2c3e50", width=3),
        )
    )

    # SARIMA forecast
    if sarima_result:
        fig.add_trace(
            go.Scatter(
                x=future_periods,
                y=sarima_result["forecast"],
                mode="lines+markers",
                name="SARIMA Forecast",
                line=dict(color="#e74c3c", width=2, dash="dash"),
            )
        )

        # Confidence intervals
        if "lower_bound" in sarima_result:
            fig.add_trace(
                go.Scatter(
                    x=future_periods,
                    y=sarima_result["upper_bound"],
                    mode="lines",
                    name="Upper Bound",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=future_periods,
                    y=sarima_result["lower_bound"],
                    mode="lines",
                    name="95% CI",
                    fill="tonexty",
                    fillcolor="rgba(231, 76, 60, 0.2)",
                    line=dict(width=0),
                )
            )

    fig.update_layout(
        title="🎯 SARIMA Revenue Forecasting",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    return fig


def detect_anomalies(ts_data, column, method="zscore", threshold=2.5):
    """Detect anomalies in time series data"""
    values = ts_data[column].values

    if method == "zscore":
        z_scores = np.abs(stats.zscore(values))
        anomalies = np.where(z_scores > threshold)[0]

    return anomalies


def create_anomaly_plot(ts_data, anomalies):
    """Create anomaly detection visualization"""
    fig = go.Figure()

    # Normal points
    normal_mask = ~ts_data.index.isin(anomalies)
    fig.add_trace(
        go.Scatter(
            x=ts_data[normal_mask]["Period"],
            y=ts_data[normal_mask]["Total_Revenue"],
            mode="lines+markers",
            name="Normal",
            line=dict(color="#2c3e50", width=2),
        )
    )

    # Anomalies
    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=ts_data.iloc[anomalies]["Period"],
                y=ts_data.iloc[anomalies]["Total_Revenue"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#e74c3c", size=10, symbol="x"),
            )
        )

    fig.update_layout(
        title="🚨 Anomaly Detection in Revenue Time Series",
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        template="plotly_white",
        height=500,
    )

    return fig


def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap"""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="📊 Feature Correlation Matrix", template="plotly_white", height=500
    )

    return fig


def create_lag_correlation_plot(ts_data, col1, col2, max_lag):
    """Create lag correlation analysis plot"""
    lags = range(0, max_lag + 1)
    correlations = []

    for lag in lags:
        if lag == 0:
            corr = ts_data[col1].corr(ts_data[col2])
        else:
            corr = ts_data[col1].iloc[lag:].corr(ts_data[col2].iloc[:-lag])
        correlations.append(corr)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(lags),
            y=correlations,
            marker_color=["#e74c3c" if lag == 0 else "#3498db" for lag in lags],
            name="Correlation",
        )
    )

    fig.update_layout(
        title=f"Lag Correlation: {col1} vs {col2}",
        xaxis_title="Lag",
        yaxis_title="Correlation Coefficient",
        template="plotly_white",
        height=400,
    )

    return fig


def generate_comprehensive_report(analyzer, ts_data, arima_result, hw_result):
    """Generate comprehensive analysis report"""
    st.success("📋 Generating comprehensive report...")

    report_content = f"""
    # 📊 Supply Chain Analytics Comprehensive Report
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## Executive Summary
    - **Analysis Period**: {ts_data['Period'].min()} to {ts_data['Period'].max()}
    - **Total Revenue**: ${ts_data['Total_Revenue'].sum():,.0f}
    - **Data Points**: {len(ts_data)} periods
    - **Forecast Horizon**: 6 periods
    
    ## Key Findings
    1. **Revenue Trends**: {((ts_data['Total_Revenue'].iloc[-1] / ts_data['Total_Revenue'].iloc[0]) - 1) * 100:.1f}% overall change
    2. **Seasonality**: {'Strong seasonal patterns detected' if len(ts_data) >= 24 else 'Insufficient data for seasonal analysis'}
    3. **Forecast Accuracy**: Multiple models deployed for robust predictions
    
    ## Model Performance
    - **ARIMA Model**: AIC = {arima_result.get('aic', 'N/A') if arima_result else 'N/A':.2f}
    - **Holt-Winters**: AIC = {hw_result.get('aic', 'N/A') if hw_result else 'N/A':.2f}
    
    ## Recommendations
    1. Monitor forecasted revenue trends for inventory planning
    2. Analyze seasonal patterns for promotional planning
    3. Review supply chain lead times for optimization opportunities
    """

    st.markdown(report_content)

    # Create downloadable report
    report_text = f"""
    Supply Chain Analytics Report
    =============================
    
    Data Summary:
    - Total Periods: {len(ts_data)}
    - Total Revenue: ${ts_data['Total_Revenue'].sum():,.0f}
    - Average Revenue per Period: ${ts_data['Total_Revenue'].mean():,.0f}
    - Revenue Volatility: {(ts_data['Total_Revenue'].std() / ts_data['Total_Revenue'].mean()) * 100:.1f}%
    
    Statistical Summary:
    {ts_data['Total_Revenue'].describe().to_string()}
    
    Forecast Results:
    - ARIMA Forecast: {arima_result['forecast'] if arima_result else 'N/A'}
    - Holt-Winters Forecast: {hw_result['forecast'] if hw_result else 'N/A'}
    """

    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name=f"supply_chain_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


def main():
    """Main application function"""

    # Sidebar for data upload
    st.sidebar.markdown(
        """
    <div class="sidebar-info">
        <h3>📁 Data Upload</h3>
        <p>Upload your supply chain CSV file to get started</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize analyzer and LLM assistant
        analyzer = SupplyChainAnalyzer("temp_data.csv")
        llm_assistant = LLMAssistant()

        # Show dashboard
        show_dashboard(analyzer, llm_assistant)
    else:
        # Show welcome message
        st.markdown(
            """
        <div style="text-align: center; padding: 5rem 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">🚀 Advanced Supply Chain Analytics</h1>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 3rem;">
                Upload your supply chain data to unlock powerful time series analysis, forecasting, and AI-driven insights
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin-top: 4rem;">
                <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3>📈 Time Series Analysis</h3>
                    <p>ARIMA, SARIMA, Holt-Winters models for accurate forecasting</p>
                </div>
                
                <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3>🤖 AI-Powered Insights</h3>
                    <p>LLM-generated summaries and actionable recommendations</p>
                </div>
                
                <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3>🔍 Advanced Analytics</h3>
                    <p>Seasonal decomposition, anomaly detection, correlation analysis</p>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Sample data structure info
        with st.expander("📋 Expected CSV Format"):
            st.markdown(
                """
            Your CSV should contain the following columns:
            - `OrderNumber`: Unique order identifier
            - `Sales Channel`: Sales channel information
            - `WarehouseCode`: Warehouse identifier
            - `ProcuredDate`: Date when items were procured
            - `OrderDate`: Date when order was placed
            - `ShipDate`: Date when order was shipped
            - `DeliveryDate`: Date when order was delivered
            - `CurrencyCode`: Currency code
            - `_SalesTeamID`: Sales team identifier
            - `_CustomerID`: Customer identifier
            - `_StoreID`: Store identifier
            - `_ProductID`: Product identifier
            - `Order Quantity`: Quantity ordered
            - `Discount Applied`: Discount amount
            - `Unit Cost`: Cost per unit
            - `Unit Price`: Price per unit
            """
            )


if __name__ == "__main__":
    main()
