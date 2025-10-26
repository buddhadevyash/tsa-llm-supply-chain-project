# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime, timedelta
# import warnings

# warnings.filterwarnings("ignore")

# # Enhanced time series forecasting imports
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import adfuller, acf, pacf
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     mean_absolute_percentage_error,
# )
# from sklearn.preprocessing import StandardScaler
# from scipy import stats

# # LLM imports
# import os
# from dotenv import load_dotenv
# import groq

# # Page configuration
# st.set_page_config(
#     page_title="Advanced Supply Chain Analytics & AI Assistant",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Load environment variables
# load_dotenv()

# # Enhanced custom CSS for modern UI
# st.markdown(
#     """
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#         font-weight: 800;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
#     .section-header {
#         font-size: 1.8rem;
#         color: #2e86ab;
#         margin-top: 2rem;
#         margin-bottom: 1rem;
#         font-weight: 700;
#         border-left: 5px solid #ff6b6b;
#         padding-left: 1.5rem;
#         background: linear-gradient(90deg, rgba(255,107,107,0.1) 0%, transparent 100%);
#         border-radius: 0 25px 25px 0;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 20px;
#         color: white;
#         text-align: center;
#         box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
#         transition: transform 0.3s ease;
#         border: 1px solid rgba(255,255,255,0.1);
#     }
#     .metric-card:hover {
#         transform: translateY(-5px);
#     }
#     .insight-box {
#         background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         border-left: 6px solid #ff6b6b;
#         margin: 1rem 0;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
#         border: 1px solid #dee2e6;
#     }
#     .forecast-card {
#         background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 15px;
#         margin: 0.5rem 0;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
#     }
#     .model-performance {
#         background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 15px;
#         margin: 0.5rem 0;
#         text-align: center;
#     }
#     .stButton button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         border-radius: 10px;
#         padding: 0.75rem 1.5rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
#     .stButton button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
#     }
#     .ai-response {
#         background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         border: 2px solid #e9ecef;
#         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
#         margin: 1rem 0;
#     }
#     .stSelectbox > div > div {
#         background-color: #f8f9fa;
#         border-radius: 10px;
#     }
#     .stDateInput > div > div {
#         background-color: #f8f9fa;
#         border-radius: 10px;
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )


# class AdvancedSupplyChainAnalyzer:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.df = None
#         self.processed_df = None
#         self.load_data()

#     def load_data(self):
#         """Load and preprocess the supply chain data with correct column mapping"""
#         try:
#             # Read the CSV file
#             self.df = pd.read_csv(self.data_path)

#             # Map to expected column names based on your specification
#             expected_columns = [
#                 "OrderNumber",
#                 "Sales_Channel",
#                 "WarehouseCode",
#                 "ProcuredDate",
#                 "OrderDate",
#                 "ShipDate",
#                 "DeliveryDate",
#                 "CurrencyCode",
#                 "SalesTeamID",
#                 "CustomerID",
#                 "StoreID",
#                 "ProductID",
#                 "Order_Quantity",
#                 "Discount_Applied",
#                 "Unit_Cost",
#                 "Unit_Price",
#             ]

#             # Assign column names if we have the right number
#             if len(self.df.columns) >= len(expected_columns):
#                 self.df = self.df.iloc[:, : len(expected_columns)]
#                 self.df.columns = expected_columns

#             # Clean and process the data
#             self.clean_data()

#         except Exception as e:
#             st.error(f"Error loading data: {e}")

#     def clean_data(self):
#         """Enhanced data cleaning and preprocessing"""
#         try:
#             # Convert date columns to datetime
#             date_columns = ["ProcuredDate", "OrderDate", "ShipDate", "DeliveryDate"]
#             for col in date_columns:
#                 if col in self.df.columns:
#                     self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

#             # Convert numeric columns
#             numeric_columns = [
#                 "Order_Quantity",
#                 "Unit_Cost",
#                 "Unit_Price",
#                 "Discount_Applied",
#             ]
#             for col in numeric_columns:
#                 if col in self.df.columns:
#                     self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

#             # Create calculated fields
#             if all(col in self.df.columns for col in ["Order_Quantity", "Unit_Price"]):
#                 self.df["Total_Revenue"] = (
#                     self.df["Order_Quantity"] * self.df["Unit_Price"]
#                 )
#                 self.df["Total_Cost"] = self.df["Order_Quantity"] * self.df["Unit_Cost"]
#                 self.df["Profit"] = self.df["Total_Revenue"] - self.df["Total_Cost"]
#                 self.df["Profit_Margin"] = (
#                     self.df["Profit"] / self.df["Total_Revenue"]
#                 ) * 100

#             # Create lead time metrics
#             if all(
#                 col in self.df.columns
#                 for col in ["OrderDate", "ShipDate", "DeliveryDate"]
#             ):
#                 self.df["Order_to_Ship_Days"] = (
#                     self.df["ShipDate"] - self.df["OrderDate"]
#                 ).dt.days
#                 self.df["Ship_to_Delivery_Days"] = (
#                     self.df["DeliveryDate"] - self.df["ShipDate"]
#                 ).dt.days
#                 self.df["Total_Lead_Time"] = (
#                     self.df["DeliveryDate"] - self.df["OrderDate"]
#                 ).dt.days

#             # Remove invalid records
#             if "OrderDate" in self.df.columns:
#                 self.df = self.df[self.df["OrderDate"].notna()]

#             # Create processed dataframe for analysis
#             self.processed_df = self.df.copy()

#         except Exception as e:
#             st.error(f"Error cleaning data: {e}")

#     def perform_stationarity_test(self, series):
#         """Perform Augmented Dickey-Fuller test for stationarity"""
#         try:
#             result = adfuller(series.dropna())
#             return {
#                 "adf_statistic": result[0],
#                 "p_value": result[1],
#                 "critical_values": result[4],
#                 "is_stationary": result[1] < 0.05,
#             }
#         except:
#             return None

#     def auto_arima_parameters(self, series, max_p=5, max_d=2, max_q=5):
#         """Automatically determine best ARIMA parameters using AIC"""
#         best_aic = np.inf
#         best_params = (0, 0, 0)

#         for p in range(max_p + 1):
#             for d in range(max_d + 1):
#                 for q in range(max_q + 1):
#                     try:
#                         model = ARIMA(series, order=(p, d, q))
#                         fitted_model = model.fit()
#                         if fitted_model.aic < best_aic:
#                             best_aic = fitted_model.aic
#                             best_params = (p, d, q)
#                     except:
#                         continue

#         return best_params, best_aic

#     def advanced_arima_forecast(self, data, column="Total_Revenue", periods=12):
#         """Enhanced ARIMA forecasting with automatic parameter selection"""
#         try:
#             series = data[column].dropna()

#             # Test for stationarity
#             stationarity_result = self.perform_stationarity_test(series)

#             # Auto-select parameters
#             best_params, best_aic = self.auto_arima_parameters(series)

#             # Fit ARIMA model
#             model = ARIMA(series, order=best_params)
#             fitted_model = model.fit()

#             # Generate forecast with confidence intervals
#             forecast_result = fitted_model.get_forecast(steps=periods)
#             forecast = forecast_result.predicted_mean
#             conf_int = forecast_result.conf_int()

#             return {
#                 "forecast": forecast.tolist(),
#                 "lower_ci": conf_int.iloc[:, 0].tolist(),
#                 "upper_ci": conf_int.iloc[:, 1].tolist(),
#                 "parameters": best_params,
#                 "aic": best_aic,
#                 "stationarity": stationarity_result,
#                 "model_summary": str(fitted_model.summary()),
#             }

#         except Exception as e:
#             st.warning(f"Advanced ARIMA forecast failed: {e}")
#             return self.fallback_forecast(data, column, periods)

#     def sarima_forecast(self, data, column="Total_Revenue", periods=12):
#         """SARIMA (Seasonal ARIMA) forecasting"""
#         try:
#             series = data[column].dropna()

#             # Try different seasonal parameters
#             seasonal_periods = min(12, len(series) // 4) if len(series) >= 24 else 0

#             if seasonal_periods > 0:
#                 model = SARIMAX(
#                     series, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_periods)
#                 )
#                 fitted_model = model.fit(disp=False)

#                 forecast_result = fitted_model.get_forecast(steps=periods)
#                 forecast = forecast_result.predicted_mean
#                 conf_int = forecast_result.conf_int()

#                 return {
#                     "forecast": forecast.tolist(),
#                     "lower_ci": conf_int.iloc[:, 0].tolist(),
#                     "upper_ci": conf_int.iloc[:, 1].tolist(),
#                     "seasonal_periods": seasonal_periods,
#                     "aic": fitted_model.aic,
#                     "model_summary": str(fitted_model.summary()),
#                 }
#             else:
#                 return self.advanced_arima_forecast(data, column, periods)

#         except Exception as e:
#             st.warning(f"SARIMA forecast failed: {e}")
#             return self.advanced_arima_forecast(data, column, periods)

#     def triple_exponential_smoothing(self, data, column="Total_Revenue", periods=12):
#         """Advanced Holt-Winters Triple Exponential Smoothing"""
#         try:
#             series = data[column].dropna()

#             # Determine seasonal periods
#             seasonal_periods = min(12, len(series) // 3) if len(series) >= 24 else None

#             if seasonal_periods:
#                 model = ExponentialSmoothing(
#                     series,
#                     trend="add",
#                     seasonal="add",
#                     seasonal_periods=seasonal_periods,
#                 )
#             else:
#                 model = ExponentialSmoothing(series, trend="add")

#             fitted_model = model.fit()
#             forecast = fitted_model.forecast(periods)

#             return {
#                 "forecast": forecast.tolist(),
#                 "seasonal_periods": seasonal_periods,
#                 "model_summary": f"Holt-Winters model with seasonal_periods={seasonal_periods}",
#             }

#         except Exception as e:
#             st.warning(f"Triple exponential smoothing failed: {e}")
#             return self.fallback_forecast(data, column, periods)

#     def calculate_advanced_metrics(self, actual, forecast):
#         """Calculate comprehensive forecast accuracy metrics"""
#         try:
#             actual = np.array(actual)
#             forecast = np.array(forecast)

#             # Ensure same length
#             min_len = min(len(actual), len(forecast))
#             actual = actual[:min_len]
#             forecast = forecast[:min_len]

#             mae = mean_absolute_error(actual, forecast)
#             mse = mean_squared_error(actual, forecast)
#             rmse = np.sqrt(mse)
#             mape = mean_absolute_percentage_error(actual, forecast) * 100

#             # Additional metrics
#             me = np.mean(forecast - actual)  # Mean Error
#             mpe = np.mean((forecast - actual) / actual) * 100  # Mean Percentage Error

#             return {
#                 "MAE": mae,
#                 "MSE": mse,
#                 "RMSE": rmse,
#                 "MAPE": mape,
#                 "ME": me,
#                 "MPE": mpe,
#             }
#         except:
#             return {}

#     def create_advanced_time_series_data(
#         self, date_range=None, channels=None, warehouses=None
#     ):
#         """Create comprehensive time series data with filtering"""
#         try:
#             df_filtered = self.processed_df.copy()

#             # Apply filters
#             if date_range:
#                 df_filtered = df_filtered[
#                     (df_filtered["OrderDate"] >= date_range[0])
#                     & (df_filtered["OrderDate"] <= date_range[1])
#                 ]

#             if channels:
#                 df_filtered = df_filtered[df_filtered["Sales_Channel"].isin(channels)]

#             if warehouses:
#                 df_filtered = df_filtered[df_filtered["WarehouseCode"].isin(warehouses)]

#             # Create multiple time series aggregations
#             time_series_data = {}

#             # Daily aggregation
#             daily_data = (
#                 df_filtered.groupby("OrderDate")
#                 .agg(
#                     {
#                         "Total_Revenue": "sum",
#                         "Order_Quantity": "sum",
#                         "OrderNumber": "count",
#                         "Profit": "sum",
#                         "Total_Lead_Time": "mean",
#                     }
#                 )
#                 .reset_index()
#             )
#             time_series_data["daily"] = daily_data

#             # Weekly aggregation
#             df_filtered["Week"] = df_filtered["OrderDate"].dt.to_period("W")
#             weekly_data = (
#                 df_filtered.groupby("Week")
#                 .agg(
#                     {
#                         "Total_Revenue": "sum",
#                         "Order_Quantity": "sum",
#                         "OrderNumber": "count",
#                         "Profit": "sum",
#                         "Total_Lead_Time": "mean",
#                     }
#                 )
#                 .reset_index()
#             )
#             weekly_data["Week"] = weekly_data["Week"].astype(str)
#             time_series_data["weekly"] = weekly_data

#             # Monthly aggregation
#             df_filtered["Month"] = df_filtered["OrderDate"].dt.to_period("M")
#             monthly_data = (
#                 df_filtered.groupby("Month")
#                 .agg(
#                     {
#                         "Total_Revenue": "sum",
#                         "Order_Quantity": "sum",
#                         "OrderNumber": "count",
#                         "Profit": "sum",
#                         "Total_Lead_Time": "mean",
#                     }
#                 )
#                 .reset_index()
#             )
#             monthly_data["Month"] = monthly_data["Month"].astype(str)
#             time_series_data["monthly"] = monthly_data

#             return time_series_data

#         except Exception as e:
#             st.error(f"Error creating time series data: {e}")
#             return {}

#     def fallback_forecast(self, data, column, periods):
#         """Fallback forecasting method"""
#         try:
#             values = data[column].values
#             last_values = values[-min(6, len(values)) :]
#             trend = np.polyfit(range(len(last_values)), last_values, 1)[0]

#             forecast = []
#             for i in range(periods):
#                 next_val = values[-1] + trend * (i + 1)
#                 forecast.append(max(next_val, 0))

#             return {"forecast": forecast}
#         except:
#             return {"forecast": [0] * periods}


# class EnhancedLLMAssistant:
#     def __init__(self):
#         self.client = None
#         self.setup_llm()

#     def setup_llm(self):
#         """Initialize Groq client"""
#         try:
#             api_key = os.getenv("GROQ_API_KEY")
#             if api_key:
#                 self.client = groq.Groq(api_key=api_key)
#             else:
#                 st.warning("GROQ_API_KEY not found in environment variables")
#         except Exception as e:
#             st.error(f"Error setting up LLM: {e}")

#     def get_structured_response(self, prompt, context="", analysis_data=None):
#         """Get structured response with data analysis"""
#         if not self.client:
#             return "LLM service not available. Please check your API key."

#         try:
#             # Enhanced context with analysis data
#             enhanced_context = f"""
#             You are an expert supply chain data scientist and analyst. Provide comprehensive, structured analysis with the following guidelines:

#             1. **Executive Summary**: Start with key findings
#             2. **Statistical Analysis**: Include relevant metrics and interpretations
#             3. **Visual Insights**: Describe what the data shows
#             4. **Recommendations**: Provide actionable insights
#             5. **Technical Details**: Include model performance when relevant

#             Supply Chain Data Context:
#             {context}

#             Analysis Results:
#             {analysis_data if analysis_data else "No additional analysis data provided"}

#             User Question: {prompt}

#             Please provide a detailed, well-structured response with clear sections, bullet points, and statistical insights.
#             """

#             response = self.client.chat.completions.create(
#                 messages=[{"role": "user", "content": enhanced_context}],
#                 model="openai/gpt-oss-120b",
#                 temperature=0.2,
#                 max_tokens=2048,
#             )

#             return response.choices[0].message.content
#         except Exception as e:
#             return f"Error getting response: {e}"

#     def generate_chart_summary(self, chart_type, data_summary):
#         """Generate AI summary for charts"""
#         if not self.client:
#             return "Chart summary not available."

#         try:
#             prompt = f"""
#             As a data visualization expert, provide a concise but insightful summary of this {chart_type} chart:

#             Data Summary: {data_summary}

#             Please provide:
#             1. Key patterns or trends visible
#             2. Notable insights or anomalies
#             3. Business implications
#             4. Recommendations based on the visualization

#             Keep the response focused and actionable (2-3 paragraphs max).
#             """

#             response = self.client.chat.completions.create(
#                 messages=[{"role": "user", "content": prompt}],
#                 model="openai/gpt-oss-120b",
#                 temperature=0.3,
#                 max_tokens=512,
#             )

#             return response.choices[0].message.content
#         except:
#             return "Unable to generate chart summary."


# def main():
#     # Enhanced sidebar with modern styling
#     st.sidebar.markdown(
#         """
#     <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
#         <h2 style="margin: 0; text-align: center; font-size: 1.5rem;">🚀 Advanced Analytics Suite</h2>
#     </div>
#     """,
#         unsafe_allow_html=True,
#     )

#     page = st.sidebar.radio(
#         "Navigation Menu",
#         [
#             "📊 Interactive Dashboard",
#             "🔮 Time Series Forecasting",
#             "🤖 AI Research Assistant",
#             "📈 Model Comparison",
#         ],
#         label_visibility="hidden",
#     )

#     # Initialize classes
#     data_path = "data/data.csv"  # Updated path
#     analyzer = AdvancedSupplyChainAnalyzer(data_path)
#     llm_assistant = EnhancedLLMAssistant()

#     if page == "📊 Interactive Dashboard":
#         show_interactive_dashboard(analyzer, llm_assistant)
#     elif page == "🔮 Time Series Forecasting":
#         show_forecasting_page(analyzer, llm_assistant)
#     elif page == "🤖 AI Research Assistant":
#         show_enhanced_research_assistant(analyzer, llm_assistant)
#     else:
#         show_model_comparison(analyzer, llm_assistant)


# def show_interactive_dashboard(analyzer, llm_assistant):
#     """Enhanced interactive dashboard with filters and AI insights"""

#     st.markdown(
#         '<div class="main-header">📊 Advanced Supply Chain Analytics Dashboard</div>',
#         unsafe_allow_html=True,
#     )

#     if analyzer.processed_df is None or analyzer.processed_df.empty:
#         st.error("No data loaded. Please check the data file path.")
#         return

#     # Interactive filters
#     st.markdown(
#         '<div class="section-header">🎛️ Interactive Data Filters</div>',
#         unsafe_allow_html=True,
#     )

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         if "OrderDate" in analyzer.processed_df.columns:
#             min_date = analyzer.processed_df["OrderDate"].min().date()
#             max_date = analyzer.processed_df["OrderDate"].max().date()
#             date_range = st.date_input(
#                 "Select Date Range",
#                 value=(min_date, max_date),
#                 min_value=min_date,
#                 max_value=max_date,
#             )

#     with col2:
#         if "Sales_Channel" in analyzer.processed_df.columns:
#             channels = st.multiselect(
#                 "Select Sales Channels",
#                 options=analyzer.processed_df["Sales_Channel"].unique(),
#                 default=analyzer.processed_df["Sales_Channel"].unique(),
#             )

#     with col3:
#         if "WarehouseCode" in analyzer.processed_df.columns:
#             warehouses = st.multiselect(
#                 "Select Warehouses",
#                 options=analyzer.processed_df["WarehouseCode"].unique(),
#                 default=analyzer.processed_df["WarehouseCode"].unique(),
#             )

#     with col4:
#         aggregation_level = st.selectbox(
#             "Time Aggregation", options=["daily", "weekly", "monthly"], index=2
#         )

#     # Get filtered time series data
#     time_series_data = analyzer.create_advanced_time_series_data(
#         date_range=date_range if "date_range" in locals() else None,
#         channels=channels if "channels" in locals() else None,
#         warehouses=warehouses if "warehouses" in locals() else None,
#     )

#     # Enhanced KPI metrics
#     st.markdown(
#         '<div class="section-header">📈 Dynamic Key Performance Indicators</div>',
#         unsafe_allow_html=True,
#     )

#     col1, col2, col3, col4, col5 = st.columns(5)

#     # Calculate filtered metrics
#     filtered_df = analyzer.processed_df.copy()
#     if "date_range" in locals() and len(date_range) == 2:
#         filtered_df = filtered_df[
#             (filtered_df["OrderDate"].dt.date >= date_range[0])
#             & (filtered_df["OrderDate"].dt.date <= date_range[1])
#         ]

#     with col1:
#         total_revenue = (
#             filtered_df["Total_Revenue"].sum()
#             if "Total_Revenue" in filtered_df.columns
#             else 0
#         )
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>💰 Total Revenue</h4>
#             <h2>${total_revenue:,.0f}</h2>
#             <small>Filtered Period</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col2:
#         total_orders = len(filtered_df)
#         st.markdown(
#             f"""
#         <div class="metric-card">
#             <h4>📦 Total Orders</h4>
#             <h2>{total_orders:,}</h2>
#             <small>Order Count</small>
#         </div>
#         """,
#             unsafe_allow_html=True,
#         )

#     with col3:
#         if "Profit" in filtered_df.columns:
#             total_profit = filtered_df["Profit"].sum()
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>💎 Total Profit</h4>
#                 <h2>${total_profit:,.0f}</h2>
#                 <small>Net Profit</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#     with col4:
#         if "Total_Lead_Time" in filtered_df.columns:
#             avg_lead_time = filtered_df["Total_Lead_Time"].mean()
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>⏱️ Avg Lead Time</h4>
#                 <h2>{avg_lead_time:.1f}</h2>
#                 <small>Days</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#     with col5:
#         if "Profit_Margin" in filtered_df.columns:
#             avg_margin = filtered_df["Profit_Margin"].mean()
#             st.markdown(
#                 f"""
#             <div class="metric-card">
#                 <h4>📊 Avg Margin</h4>
#                 <h2>{avg_margin:.1f}%</h2>
#                 <small>Profit Margin</small>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#     # Advanced visualizations with AI summaries
#     st.markdown(
#         '<div class="section-header">📊 Advanced Analytics with AI Insights</div>',
#         unsafe_allow_html=True,
#     )

#     if (
#         aggregation_level in time_series_data
#         and not time_series_data[aggregation_level].empty
#     ):
#         data = time_series_data[aggregation_level]

#         col1, col2 = st.columns([2, 1])

#         with col1:
#             # Revenue trend with enhanced styling
#             time_col = (
#                 "OrderDate"
#                 if aggregation_level == "daily"
#                 else ("Week" if aggregation_level == "weekly" else "Month")
#             )

#             fig = px.line(
#                 data,
#                 x=time_col,
#                 y="Total_Revenue",
#                 title=f"📈 Revenue Trend - {aggregation_level.title()} View",
#                 template="plotly_white",
#             )
#             fig.update_traces(line=dict(width=4, color="#667eea"))
#             fig.update_layout(
#                 height=400,
#                 plot_bgcolor="rgba(0,0,0,0)",
#                 paper_bgcolor="rgba(0,0,0,0)",
#                 font=dict(size=12),
#                 title_font_size=16,
#             )
#             st.plotly_chart(fig, use_container_width=True)

#         with col2:
#             # AI-generated chart summary
#             st.markdown("### 🤖 AI Chart Analysis")
#             data_summary = f"""
#             Revenue trend over {len(data)} {aggregation_level} periods.
#             Total revenue: ${data["Total_Revenue"].sum():,.0f}
#             Average: ${data["Total_Revenue"].mean():,.0f}
#             Max: ${data["Total_Revenue"].max():,.0f}
#             Min: ${data["Total_Revenue"].min():,.0f}
#             Trend: {"Increasing" if data["Total_Revenue"].iloc[-1] > data["Total_Revenue"].iloc[0] else "Decreasing"}
#             """

#             chart_summary = llm_assistant.generate_chart_summary(
#                 "Revenue Trend", data_summary
#             )
#             st.markdown(
#                 f'<div class="ai-response">{chart_summary}</div>',
#                 unsafe_allow_html=True,
#             )

#     # Multi-metric dashboard
#     col1, col2 = st.columns(2)

#     with col1:
#         if aggregation_level in time_series_data:
#             data = time_series_data[aggregation_level]
#             time_col = (
#                 "OrderDate"
#                 if aggregation_level == "daily"
#                 else ("Week" if aggregation_level == "weekly" else "Month")
#             )

#             # Multi-line chart
#             fig = go.Figure()
#             fig.add_trace(
#                 go.Scatter(
#                     x=data[time_col],
#                     y=data["Total_Revenue"],
#                     name="Revenue",
#                     line=dict(color="#667eea", width=3),
#                 )
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=data[time_col],
#                     y=data["Profit"],
#                     name="Profit",
#                     line=dict(color="#ff6b6b", width=3),
#                 )
#             )

#             fig.update_layout(
#                 title="💰 Revenue vs Profit Analysis",
#                 template="plotly_white",
#                 height=400,
#                 plot_bgcolor="rgba(0,0,0,0)",
#                 paper_bgcolor="rgba(0,0,0,0)",
#             )
#             st.plotly_chart(fig, use_container_width=True)

#     with col2:
#         # Order quantity distribution
#         if "Order_Quantity" in filtered_df.columns:
#             fig = px.histogram(
#                 filtered_df,
#                 x="Order_Quantity",
#                 nbins=30,
#                 title="📦 Order Quantity Distribution",
#                 template="plotly_white",
#             )
#             fig.update_traces(marker_color="#74b9ff")
#             fig.update_layout(height=400)
#             st.plotly_chart(fig, use_container_width=True)


# def show_forecasting_page(analyzer, llm_assistant):
#     """Advanced time series forecasting page"""

#     st.markdown(
#         '<div class="main-header">🔮 Advanced Time Series Forecasting</div>',
#         unsafe_allow_html=True,
#     )

#     if analyzer.processed_df is None or analyzer.processed_df.empty:
#         st.error("No data loaded for forecasting.")
#         return

#     # Forecasting controls
#     st.markdown(
#         '<div class="section-header">⚙️ Forecasting Configuration</div>',
#         unsafe_allow_html=True,
#     )

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         forecast_periods = st.slider("Forecast Periods", 3, 24, 12)

#     with col2:
#         target_metric = st.selectbox(
#             "Target Metric", ["Total_Revenue", "Order_Quantity", "Profit"]
#         )

#     with col3:
#         aggregation = st.selectbox("Time Aggregation", ["monthly", "weekly"], index=0)

#     with col4:
#         if st.button("🚀 Generate Forecasts", type="primary"):
#             st.session_state.run_forecast = True

#     # Get time series data
#     time_series_data = analyzer.create_advanced_time_series_data()

#     if aggregation in time_series_data and not time_series_data[aggregation].empty:
#         data = time_series_data[aggregation]

#         if "run_forecast" in st.session_state and st.session_state.run_forecast:
#             # Generate multiple forecasts
#             with st.spinner("🔬 Running advanced forecasting models..."):
#                 arima_result = analyzer.advanced_arima_forecast(
#                     data, target_metric, forecast_periods
#                 )
#                 sarima_result = analyzer.sarima_forecast(
#                     data, target_metric, forecast_periods
#                 )
#                 holt_result = analyzer.triple_exponential_smoothing(
#                     data, target_metric, forecast_periods
#                 )

#             # Forecast visualization
#             st.markdown(
#                 '<div class="section-header">📊 Forecast Results Comparison</div>',
#                 unsafe_allow_html=True,
#             )

#             fig = go.Figure()

#             # Historical data
#             time_col = "Week" if aggregation == "weekly" else "Month"
#             fig.add_trace(
#                 go.Scatter(
#                     x=data[time_col],
#                     y=data[target_metric],
#                     name="Historical Data",
#                     line=dict(width=4, color="#1f77b4"),
#                 )
#             )

#             # Forecast periods
#             last_period = data[time_col].iloc[-1]
#             if aggregation == "monthly":
#                 forecast_periods_labels = [f"F{i + 1}" for i in range(forecast_periods)]
#             else:
#                 forecast_periods_labels = [f"W{i + 1}" for i in range(forecast_periods)]

#             # ARIMA forecast
#             if arima_result and "forecast" in arima_result:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=forecast_periods_labels,
#                         y=arima_result["forecast"],
#                         name=f"ARIMA{arima_result.get('parameters', '')}",
#                         line=dict(width=3, color="#ff6b6b", dash="dash"),
#                     )
#                 )

#                 if "upper_ci" in arima_result and "lower_ci" in arima_result:
#                     fig.add_trace(
#                         go.Scatter(
#                             x=forecast_periods_labels,
#                             y=arima_result["upper_ci"],
#                             fill=None,
#                             mode="lines",
#                             line_color="rgba(255,107,107,0)",
#                             showlegend=False,
#                         )
#                     )
#                     fig.add_trace(
#                         go.Scatter(
#                             x=forecast_periods_labels,
#                             y=arima_result["lower_ci"],
#                             fill="tonexty",
#                             mode="lines",
#                             line_color="rgba(255,107,107,0)",
#                             name="ARIMA Confidence Interval",
#                             fillcolor="rgba(255,107,107,0.2)",
#                         )
#                     )

#             # SARIMA forecast
#             if sarima_result and "forecast" in sarima_result:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=forecast_periods_labels,
#                         y=sarima_result["forecast"],
#                         name="SARIMA",
#                         line=dict(width=3, color="#2ecc71", dash="dash"),
#                     )
#                 )

#             # Holt-Winters forecast
#             if holt_result and "forecast" in holt_result:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=forecast_periods_labels,
#                         y=holt_result["forecast"],
#                         name="Holt-Winters",
#                         line=dict(width=3, color="#f39c12", dash="dash"),
#                     )
#                 )

#             fig.update_layout(
#                 title=f"🔮 {target_metric} Forecasting - Multiple Models Comparison",
#                 template="plotly_white",
#                 height=600,
#                 plot_bgcolor="rgba(0,0,0,0)",
#                 paper_bgcolor="rgba(0,0,0,0)",
#                 font=dict(size=12),
#             )

#             st.plotly_chart(fig, use_container_width=True)

#             # Model performance and insights
#             st.markdown(
#                 '<div class="section-header">📊 Model Performance & Statistical Analysis</div>',
#                 unsafe_allow_html=True,
#             )

#             col1, col2, col3 = st.columns(3)

#             with col1:
#                 if arima_result is not None:
#                     aic_value = arima_result.get("aic", "N/A")
#                     aic_display = (
#                         f"{aic_value:.2f}"
#                         if isinstance(aic_value, (int, float))
#                         else "N/A"
#                     )

#                     st.markdown(
#                         f"""
#                                 <div class="model-performance">
#                                     <h4>🔄 ARIMA Model</h4>
#                                     <p><strong>Parameters:</strong> {arima_result.get("parameters", "N/A")}</p>
#                                     <p><strong>AIC:</strong> {aic_display}</p>
#                                     <p><strong>Stationarity:</strong> {"✅" if arima_result.get("stationarity", {}).get("is_stationary", False) else "❌"}</p>
#                                 </div>
#                                 """,
#                         unsafe_allow_html=True,
#                     )
#                 else:
#                     st.markdown(
#                         """
#                                 <div class="model-performance">
#                                     <h4>🔄 ARIMA Model</h4>
#                                     <p><strong>Status:</strong> ❌ Model failed</p>
#                                     <p><strong>Error:</strong> Unable to fit ARIMA model</p>
#                                     <p><strong>Fallback:</strong> Using simple forecast</p>
#                                 </div>
#                                 """,
#                         unsafe_allow_html=True,
#                     )

#             with col2:
#                 if sarima_result is not None:
#                     aic_value = sarima_result.get("aic", "N/A")
#                     aic_display = (
#                         f"{aic_value:.2f}"
#                         if isinstance(aic_value, (int, float))
#                         else "N/A"
#                     )

#                     st.markdown(
#                         f"""
#                   <div class="model-performance">
#                       <h4>📈 SARIMA Model</h4>
#                       <p><strong>Seasonal Periods:</strong> {sarima_result.get("seasonal_periods", "N/A")}</p>
#                       <p><strong>AIC:</strong> {aic_display}</p>
#                       <p><strong>Seasonality:</strong> {"✅" if sarima_result.get("seasonal_periods", 0) > 0 else "❌"}</p>
#                   </div>
#                   """,
#                         unsafe_allow_html=True,
#                     )
#                 else:
#                     st.markdown(
#                         """
#                   <div class="model-performance">
#                       <h4>📈 SARIMA Model</h4>
#                       <p><strong>Status:</strong> ❌ Model failed</p>
#                       <p><strong>Error:</strong> Unable to fit SARIMA model</p>
#                       <p><strong>Fallback:</strong> Using simple forecast</p>
#                   </div>
#                   """,
#                         unsafe_allow_html=True,
#                     )

#             with col3:
#                 if holt_result is not None:
#                     st.markdown(
#                         f"""
#                                 <div class="model-performance">
#                                     <h4>🌊 Holt-Winters</h4>
#                                     <p><strong>Seasonal Periods:</strong> {holt_result.get("seasonal_periods", "N/A")}</p>
#                                     <p><strong>Method:</strong> Triple Exponential</p>
#                                     <p><strong>Components:</strong> Trend + Seasonal</p>
#                                 </div>
#                                 """,
#                         unsafe_allow_html=True,
#                     )
#                 else:
#                     st.markdown(
#                         """
#                                 <div class="model-performance">
#                                     <h4>🌊 Holt-Winters</h4>
#                                     <p><strong>Status:</strong> ❌ Model failed</p>
#                                     <p><strong>Error:</strong> Unable to fit Holt-Winters model</p>
#                                     <p><strong>Fallback:</strong> Using simple forecast</p>
#                                 </div>
#                                 """,
#                         unsafe_allow_html=True,
#                     )

#             # AI-generated forecast analysis
#             st.markdown(
#                 '<div class="section-header">🤖 AI Forecast Analysis</div>',
#                 unsafe_allow_html=True,
#             )

#             forecast_context = f"""
#                         Forecasting Analysis for {target_metric}:
#                         - Historical data points: {len(data)}
#                         - Forecast periods: {forecast_periods}
#                         - ARIMA forecast average: ${np.mean(arima_result.get("forecast", [0]) if arima_result else [0]):,.0f}
#                         - SARIMA forecast average: ${np.mean(sarima_result.get("forecast", [0]) if sarima_result else [0]):,.0f}
#                         - Holt-Winters forecast average: ${np.mean(holt_result.get("forecast", [0]) if holt_result else [0]):,.0f}
#                         - Best ARIMA parameters: {arima_result.get("parameters", "N/A") if arima_result else "N/A"}
#                         - Stationarity test p-value: {arima_result.get("stationarity", {}).get("p_value", "N/A") if arima_result else "N/A"}
#                         """

#             forecast_analysis = llm_assistant.get_structured_response(
#                 "Analyze these time series forecasting results and provide insights on model performance, forecast reliability, and business recommendations.",
#                 forecast_context,
#             )

#             st.markdown(
#                 f'<div class="ai-response">{forecast_analysis}</div>',
#                 unsafe_allow_html=True,
#             )


# def show_enhanced_research_assistant(analyzer, llm_assistant):
#     """Enhanced AI research assistant with advanced capabilities"""

#     st.markdown(
#         '<div class="main-header">🤖 Advanced AI Research Assistant</div>',
#         unsafe_allow_html=True,
#     )

#     st.markdown(
#         """
#     <div class="insight-box">
#     <h3>🎯 Advanced Analytics Capabilities</h3>
#     <p>This AI assistant combines statistical analysis, time series modeling, and business intelligence to provide
#     comprehensive insights into your supply chain data. It can perform advanced forecasting, identify patterns,
#     and provide actionable recommendations.</p>
#     </div>
#     """,
#         unsafe_allow_html=True,
#     )

#     # Enhanced context generation
#     context = generate_comprehensive_context(analyzer)

#     # Advanced research questions
#     st.markdown("### 💡 Advanced Research Questions")

#     advanced_questions = [
#         "Perform comprehensive time series decomposition and identify seasonal patterns with statistical significance",
#         "Compare ARIMA vs SARIMA model performance for inventory optimization and provide implementation roadmap",
#         "Analyze supply chain variability using statistical process control and recommend improvement strategies",
#         "Evaluate the impact of different sales channels on forecasting accuracy and profitability",
#         "Conduct advanced correlation analysis between lead times, order patterns, and customer satisfaction metrics",
#         "Design an optimal inventory management strategy using time series forecasting and economic order quantity principles",
#     ]

#     col1, col2 = st.columns(2)

#     for i, question in enumerate(advanced_questions):
#         col = col1 if i % 2 == 0 else col2
#         with col:
#             if st.button(f"🎯 {question[:50]}...", key=f"adv_q_{i}"):
#                 st.session_state.research_question = question

#     # Research question input
#     st.markdown("### 🔍 Ask an Advanced Research Question")
#     research_question = st.text_area(
#         "Enter your detailed research question:",
#         value=st.session_state.get("research_question", ""),
#         height=120,
#         placeholder="e.g., Using advanced time series analysis, evaluate the seasonal impact on inventory levels and develop a predictive model for optimal stock management...",
#     )

#     col1, col2 = st.columns([3, 1])

#     with col1:
#         if st.button("🚀 Generate Advanced AI Analysis", type="primary"):
#             if research_question:
#                 with st.spinner(
#                     "🔬 Performing comprehensive analysis with advanced AI models..."
#                 ):
#                     # Generate additional analysis data
#                     analysis_data = generate_analysis_data(analyzer)

#                     response = llm_assistant.get_structured_response(
#                         research_question, context, analysis_data
#                     )

#                     st.markdown("### 📋 Comprehensive AI Analysis")
#                     st.markdown(
#                         f'<div class="ai-response">{response}</div>',
#                         unsafe_allow_html=True,
#                     )

#     with col2:
#         st.markdown("### 🔧 Analysis Tools")
#         if st.button("📊 Generate Data Summary"):
#             summary = generate_data_summary(analyzer)
#             st.markdown(
#                 f'<div class="insight-box">{summary}</div>', unsafe_allow_html=True
#             )


# def show_model_comparison(analyzer, llm_assistant):
#     """Model comparison and evaluation page"""

#     st.markdown(
#         '<div class="main-header">📈 Advanced Model Comparison & Evaluation</div>',
#         unsafe_allow_html=True,
#     )

#     if analyzer.processed_df is None or analyzer.processed_df.empty:
#         st.error("No data loaded for model comparison.")
#         return

#     st.markdown(
#         """
#     <div class="insight-box">
#     <h3>🎯 Model Evaluation Framework</h3>
#     <p>Compare different time series forecasting models using comprehensive statistical metrics
#     and cross-validation techniques to determine the best approach for your supply chain data.</p>
#     </div>
#     """,
#         unsafe_allow_html=True,
#     )

#     # Model comparison interface would go here
#     st.markdown("### 🔄 Model Comparison Results")
#     st.info(
#         "This section will compare multiple forecasting models and provide detailed performance metrics."
#     )


# def generate_comprehensive_context(analyzer):
#     """Generate comprehensive context for AI analysis"""
#     if analyzer.processed_df is None:
#         return "No data available for analysis."

#     context = f"""
#     ## Comprehensive Supply Chain Dataset Analysis

#     ### Dataset Overview:
#     - Total Records: {len(analyzer.processed_df):,}
#     - Date Range: {analyzer.processed_df["OrderDate"].min().strftime("%Y-%m-%d") if "OrderDate" in analyzer.processed_df.columns else "N/A"} to {analyzer.processed_df["OrderDate"].max().strftime("%Y-%m-%d") if "OrderDate" in analyzer.processed_df.columns else "N/A"}

#     ### Financial Metrics:
#     - Total Revenue: ${analyzer.processed_df["Total_Revenue"].sum():,.2f if 'Total_Revenue' in analyzer.processed_df.columns else 0}
#     - Average Order Value: ${analyzer.processed_df["Total_Revenue"].mean():,.2f if 'Total_Revenue' in analyzer.processed_df.columns else 0}
#     - Total Profit: ${analyzer.processed_df["Profit"].sum():,.2f if 'Profit' in analyzer.processed_df.columns else 0}
#     - Average Profit Margin: {analyzer.processed_df["Profit_Margin"].mean():.2f}% if 'Profit_Margin' in analyzer.processed_df.columns else 0%

#     ### Operational Metrics:
#     - Average Lead Time: {analyzer.processed_df["Total_Lead_Time"].mean():.1f} days if 'Total_Lead_Time' in analyzer.processed_df.columns else 'N/A'
#     - Sales Channels: {analyzer.processed_df["Sales_Channel"].nunique() if "Sales_Channel" in analyzer.processed_df.columns else 0}
#     - Warehouses: {analyzer.processed_df["WarehouseCode"].nunique() if "WarehouseCode" in analyzer.processed_df.columns else 0}
#     """

#     return context


# def generate_analysis_data(analyzer):
#     """Generate additional analysis data for AI context"""
#     if analyzer.processed_df is None:
#         return "No analysis data available."

#     try:
#         time_series_data = analyzer.create_advanced_time_series_data()
#         if "monthly" in time_series_data and not time_series_data["monthly"].empty:
#             monthly_data = time_series_data["monthly"]

#             analysis = f"""
#             Time Series Analysis Results:
#             - Monthly data points: {len(monthly_data)}
#             - Revenue trend: {"Upward" if monthly_data["Total_Revenue"].iloc[-1] > monthly_data["Total_Revenue"].iloc[0] else "Downward"}
#             - Revenue volatility (CV): {monthly_data["Total_Revenue"].std() / monthly_data["Total_Revenue"].mean():.2f}
#             - Seasonal patterns detected: {len(monthly_data) >= 12}
#             """
#             return analysis
#     except:
#         pass

#     return "Basic analysis data available."


# def generate_data_summary(analyzer):
#     """Generate a comprehensive data summary"""
#     if analyzer.processed_df is None:
#         return "No data available for summary."

#     df = analyzer.processed_df

#     summary = f"""
#     <h4>📊 Comprehensive Data Summary</h4>
#     <ul>
#         <li><strong>Total Records:</strong> {len(df):,}</li>
#         <li><strong>Date Coverage:</strong> {(df["OrderDate"].max() - df["OrderDate"].min()).days if "OrderDate" in df.columns else "N/A"} days</li>
#         <li><strong>Missing Values:</strong> {df.isnull().sum().sum()} total</li>
#         <li><strong>Unique Customers:</strong> {df["CustomerID"].nunique() if "CustomerID" in df.columns else "N/A"}</li>
#         <li><strong>Unique Products:</strong> {df["ProductID"].nunique() if "ProductID" in df.columns else "N/A"}</li>
#         <li><strong>Revenue Distribution:</strong> Mean ${df["Total_Revenue"].mean():,.0f if 'Total_Revenue' in df.columns else 0}, Std ${df["Total_Revenue"].std():,.0f if 'Total_Revenue' in df.columns else 0}</li>
#     </ul>
#     """

#     return summary


# if __name__ == "__main__":
#     # Initialize session state
#     if "research_question" not in st.session_state:
#         st.session_state.research_question = ""
#     if "run_forecast" not in st.session_state:
#         st.session_state.run_forecast = False

#     main()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Enhanced time series forecasting imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

# LLM imports
import os
from dotenv import load_dotenv
import groq

# Page configuration
st.set_page_config(
    page_title="Advanced Supply Chain Analytics & AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment variables
load_dotenv()

# Enhanced custom CSS for modern UI
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
        border-left: 5px solid #ff6b6b;
        padding-left: 1.5rem;
        background: linear-gradient(90deg, rgba(255,107,107,0.1) 0%, transparent 100%);
        border-radius: 0 25px 25px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #ff6b6b;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #dee2e6;
    }
    .forecast-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .model-performance {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .ai-response {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .stDateInput > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


class AdvancedSupplyChainAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.load_data()

    def load_data(self):
        """Load and preprocess the supply chain data with correct column mapping"""
        try:
            # Read the CSV file
            self.df = pd.read_csv(self.data_path)

            # Map to expected column names based on your specification
            expected_columns = [
                "OrderNumber",
                "Sales_Channel",
                "WarehouseCode",
                "ProcuredDate",
                "OrderDate",
                "ShipDate",
                "DeliveryDate",
                "CurrencyCode",
                "SalesTeamID",
                "CustomerID",
                "StoreID",
                "ProductID",
                "Order_Quantity",
                "Discount_Applied",
                "Unit_Cost",
                "Unit_Price",
            ]

            # Assign column names if we have the right number
            if len(self.df.columns) >= len(expected_columns):
                self.df = self.df.iloc[:, : len(expected_columns)]
                self.df.columns = expected_columns

            # Clean and process the data
            self.clean_data()

        except Exception as e:
            st.error(f"Error loading data: {e}")

    def clean_data(self):
        """Enhanced data cleaning and preprocessing"""
        try:
            # Convert date columns to datetime
            date_columns = ["ProcuredDate", "OrderDate", "ShipDate", "DeliveryDate"]
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

            # Create calculated fields
            if all(col in self.df.columns for col in ["Order_Quantity", "Unit_Price"]):
                self.df["Total_Revenue"] = (
                    self.df["Order_Quantity"] * self.df["Unit_Price"]
                )
                self.df["Total_Cost"] = self.df["Order_Quantity"] * self.df["Unit_Cost"]
                self.df["Profit"] = self.df["Total_Revenue"] - self.df["Total_Cost"]
                self.df["Profit_Margin"] = (
                    self.df["Profit"] / self.df["Total_Revenue"]
                ) * 100

            # Create lead time metrics
            if all(
                col in self.df.columns
                for col in ["OrderDate", "ShipDate", "DeliveryDate"]
            ):
                self.df["Order_to_Ship_Days"] = (
                    self.df["ShipDate"] - self.df["OrderDate"]
                ).dt.days
                self.df["Ship_to_Delivery_Days"] = (
                    self.df["DeliveryDate"] - self.df["ShipDate"]
                ).dt.days
                self.df["Total_Lead_Time"] = (
                    self.df["DeliveryDate"] - self.df["OrderDate"]
                ).dt.days

            # Remove invalid records
            if "OrderDate" in self.df.columns:
                self.df = self.df[self.df["OrderDate"].notna()]

            # Create processed dataframe for analysis
            self.processed_df = self.df.copy()

        except Exception as e:
            st.error(f"Error cleaning data: {e}")

    def perform_stationarity_test(self, series):
        """Perform Augmented Dickey-Fuller test for stationarity"""
        try:
            result = adfuller(series.dropna())
            return {
                "adf_statistic": result[0],
                "p_value": result[1],
                "critical_values": result[4],
                "is_stationary": result[1] < 0.05,
            }
        except:
            return None

    def auto_arima_parameters(self, series, max_p=3, max_d=2, max_q=3):
        """Automatically determine best ARIMA parameters using AIC"""
        best_aic = np.inf
        best_params = (1, 1, 1)  # Default fallback

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                    except:
                        continue

        return best_params, best_aic

    def advanced_arima_forecast(self, data, column="Total_Revenue", periods=12):
        """Enhanced ARIMA forecasting with automatic parameter selection"""
        try:
            series = data[column].dropna()
            
            if len(series) < 10:
                return self.fallback_forecast(data, column, periods)

            # Test for stationarity
            stationarity_result = self.perform_stationarity_test(series)

            # Auto-select parameters
            best_params, best_aic = self.auto_arima_parameters(series)

            # Fit ARIMA model
            model = ARIMA(series, order=best_params)
            fitted_model = model.fit()

            # Generate forecast with confidence intervals
            forecast_result = fitted_model.get_forecast(steps=periods)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()

            return {
                "forecast": forecast.tolist(),
                "lower_ci": conf_int.iloc[:, 0].tolist(),
                "upper_ci": conf_int.iloc[:, 1].tolist(),
                "parameters": best_params,
                "aic": best_aic,
                "stationarity": stationarity_result,
                "model_summary": str(fitted_model.summary()),
            }

        except Exception as e:
            st.warning(f"Advanced ARIMA forecast failed: {e}")
            return self.fallback_forecast(data, column, periods)

    def sarima_forecast(self, data, column="Total_Revenue", periods=12):
        """SARIMA (Seasonal ARIMA) forecasting"""
        try:
            series = data[column].dropna()
            
            if len(series) < 24:  # Need sufficient data for seasonal analysis
                return self.advanced_arima_forecast(data, column, periods)

            # Try different seasonal parameters
            seasonal_periods = min(12, len(series) // 4) if len(series) >= 24 else 0

            if seasonal_periods > 0:
                model = SARIMAX(
                    series, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_periods)
                )
                fitted_model = model.fit(disp=False)

                forecast_result = fitted_model.get_forecast(steps=periods)
                forecast = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int()

                return {
                    "forecast": forecast.tolist(),
                    "lower_ci": conf_int.iloc[:, 0].tolist(),
                    "upper_ci": conf_int.iloc[:, 1].tolist(),
                    "seasonal_periods": seasonal_periods,
                    "aic": fitted_model.aic,
                    "model_summary": str(fitted_model.summary()),
                }
            else:
                return self.advanced_arima_forecast(data, column, periods)

        except Exception as e:
            st.warning(f"SARIMA forecast failed: {e}")
            return self.advanced_arima_forecast(data, column, periods)

    def triple_exponential_smoothing(self, data, column="Total_Revenue", periods=12):
        """Advanced Holt-Winters Triple Exponential Smoothing"""
        try:
            series = data[column].dropna()
            
            if len(series) < 10:
                return self.fallback_forecast(data, column, periods)

            # Determine seasonal periods
            seasonal_periods = min(12, len(series) // 3) if len(series) >= 24 else None

            if seasonal_periods:
                model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                )
            else:
                model = ExponentialSmoothing(series, trend="add")

            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)

            return {
                "forecast": forecast.tolist(),
                "seasonal_periods": seasonal_periods,
                "model_summary": f"Holt-Winters model with seasonal_periods={seasonal_periods}",
            }

        except Exception as e:
            st.warning(f"Triple exponential smoothing failed: {e}")
            return self.fallback_forecast(data, column, periods)

    def calculate_advanced_metrics(self, actual, forecast):
        """Calculate comprehensive forecast accuracy metrics"""
        try:
            actual = np.array(actual)
            forecast = np.array(forecast)

            # Ensure same length
            min_len = min(len(actual), len(forecast))
            actual = actual[:min_len]
            forecast = forecast[:min_len]

            mae = mean_absolute_error(actual, forecast)
            mse = mean_squared_error(actual, forecast)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(actual, forecast) * 100

            # Additional metrics
            me = np.mean(forecast - actual)  # Mean Error
            mpe = np.mean((forecast - actual) / actual) * 100  # Mean Percentage Error

            return {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "ME": me,
                "MPE": mpe,
            }
        except:
            return {}

    def create_advanced_time_series_data(
        self, date_range=None, channels=None, warehouses=None
    ):
        """Create comprehensive time series data with filtering"""
        try:
            df_filtered = self.processed_df.copy()

            # Apply filters with proper datetime handling
            if date_range and len(date_range) == 2:
                # Convert date objects to datetime for comparison
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                df_filtered = df_filtered[
                    (df_filtered["OrderDate"] >= start_date)
                    & (df_filtered["OrderDate"] <= end_date)
                ]

            if channels and len(channels) > 0:
                df_filtered = df_filtered[df_filtered["Sales_Channel"].isin(channels)]

            if warehouses and len(warehouses) > 0:
                df_filtered = df_filtered[df_filtered["WarehouseCode"].isin(warehouses)]

            # Create multiple time series aggregations
            time_series_data = {}

            # Daily aggregation
            if len(df_filtered) > 0:
                daily_data = (
                    df_filtered.groupby("OrderDate")
                    .agg(
                        {
                            "Total_Revenue": "sum",
                            "Order_Quantity": "sum",
                            "OrderNumber": "count",
                            "Profit": "sum",
                            "Total_Lead_Time": "mean",
                        }
                    )
                    .reset_index()
                )
                time_series_data["daily"] = daily_data

                # Weekly aggregation
                df_filtered["Week"] = df_filtered["OrderDate"].dt.to_period("W")
                weekly_data = (
                    df_filtered.groupby("Week")
                    .agg(
                        {
                            "Total_Revenue": "sum",
                            "Order_Quantity": "sum",
                            "OrderNumber": "count",
                            "Profit": "sum",
                            "Total_Lead_Time": "mean",
                        }
                    )
                    .reset_index()
                )
                weekly_data["Week"] = weekly_data["Week"].astype(str)
                time_series_data["weekly"] = weekly_data

                # Monthly aggregation
                df_filtered["Month"] = df_filtered["OrderDate"].dt.to_period("M")
                monthly_data = (
                    df_filtered.groupby("Month")
                    .agg(
                        {
                            "Total_Revenue": "sum",
                            "Order_Quantity": "sum",
                            "OrderNumber": "count",
                            "Profit": "sum",
                            "Total_Lead_Time": "mean",
                        }
                    )
                    .reset_index()
                )
                monthly_data["Month"] = monthly_data["Month"].astype(str)
                time_series_data["monthly"] = monthly_data

            return time_series_data

        except Exception as e:
            st.error(f"Error creating time series data: {e}")
            return {}

    def fallback_forecast(self, data, column, periods):
        """Fallback forecasting method"""
        try:
            values = data[column].values
            if len(values) == 0:
                return {"forecast": [0] * periods}
                
            last_values = values[-min(6, len(values)) :]
            
            if len(last_values) > 1:
                trend = np.polyfit(range(len(last_values)), last_values, 1)[0]
            else:
                trend = 0

            forecast = []
            for i in range(periods):
                next_val = values[-1] + trend * (i + 1)
                forecast.append(max(next_val, 0))

            return {"forecast": forecast}
        except:
            return {"forecast": [0] * periods}


class EnhancedLLMAssistant:
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
                st.warning("GROQ_API_KEY not found in environment variables")
        except Exception as e:
            st.error(f"Error setting up LLM: {e}")

    def get_structured_response(self, prompt, context="", analysis_data=None):
        """Get structured response with data analysis"""
        if not self.client:
            return "LLM service not available. Please check your API key."

        try:
            # Enhanced context with analysis data
            enhanced_context = f"""
            You are an expert supply chain data scientist and analyst. Provide comprehensive, structured analysis with the following guidelines:

            1. **Executive Summary**: Start with key findings
            2. **Statistical Analysis**: Include relevant metrics and interpretations
            3. **Visual Insights**: Describe what the data shows
            4. **Recommendations**: Provide actionable insights
            5. **Technical Details**: Include model performance when relevant

            Supply Chain Data Context:
            {context}

            Analysis Results:
            {analysis_data if analysis_data else "No additional analysis data provided"}

            User Question: {prompt}

            Please provide a detailed, well-structured response with clear sections, bullet points, and statistical insights.
            """

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": enhanced_context}],
                model="openai/gpt-oss-120b",
                temperature=0.2,
                max_tokens=2048,
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting response: {e}"

    def generate_chart_summary(self, chart_type, data_summary):
        """Generate AI summary for charts"""
        if not self.client:
            return "Chart summary not available."

        try:
            prompt = f"""
            As a data visualization expert, provide a concise but insightful summary of this {chart_type} chart:

            Data Summary: {data_summary}

            Please provide:
            1. Key patterns or trends visible
            2. Notable insights or anomalies
            3. Business implications
            4. Recommendations based on the visualization

            Keep the response focused and actionable (2-3 paragraphs max).
            """

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.3,
                max_tokens=512,
            )

            return response.choices[0].message.content
        except:
            return "Unable to generate chart summary."


def main():
    # Enhanced sidebar with modern styling
    st.sidebar.markdown(
        """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
        <h2 style="margin: 0; text-align: center; font-size: 1.5rem;">🚀 Advanced Analytics Suite</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "📊 Interactive Dashboard",
            "🔮 Time Series Forecasting",
            "🤖 AI Research Assistant",
            "📈 Model Comparison",
        ],
        label_visibility="hidden",
    )

    # Initialize classes
    data_path = "data/data.csv"  # Updated path
    analyzer = AdvancedSupplyChainAnalyzer(data_path)
    llm_assistant = EnhancedLLMAssistant()

    if page == "📊 Interactive Dashboard":
        show_interactive_dashboard(analyzer, llm_assistant)
    elif page == "🔮 Time Series Forecasting":
        show_forecasting_page(analyzer, llm_assistant)
    elif page == "🤖 AI Research Assistant":
        show_enhanced_research_assistant(analyzer, llm_assistant)
    else:
        show_model_comparison(analyzer, llm_assistant)


def generate_revenue_insights(df, time_series_data=None):
    """Generate AI-powered revenue insights"""
    insights = []
    
    if len(df) == 0:
        return "<p>No data available for analysis.</p>"
    
    # Revenue distribution analysis
    if 'Total_Revenue' in df.columns:
        total_revenue = df['Total_Revenue'].sum()
        avg_revenue = df['Total_Revenue'].mean()
        median_revenue = df['Total_Revenue'].median()
        
        insights.append(f"<p>📈 <strong>Total Revenue:</strong> ${total_revenue:,.0f}</p>")
        insights.append(f"<p>💡 <strong>Average Order Value:</strong> ${avg_revenue:,.0f}</p>")
        
        if avg_revenue > median_revenue * 1.2:
            insights.append("<p>⚠️ High-value orders are driving revenue (right-skewed distribution)</p>")
        
        # Growth trend analysis
        if time_series_data is not None and len(time_series_data) > 3:
            recent_period = time_series_data['Total_Revenue'].tail(3).mean()
            earlier_period = time_series_data['Total_Revenue'].head(3).mean()
            if earlier_period > 0:
                growth = ((recent_period - earlier_period) / earlier_period) * 100
                if growth > 5:
                    insights.append(f"<p>🚀 <strong>Positive trend:</strong> {growth:+.1f}% growth detected</p>")
                elif growth < -5:
                    insights.append(f"<p>📉 <strong>Declining trend:</strong> {growth:+.1f}% change detected</p>")
    
    # Channel performance
    if 'Sales_Channel' in df.columns:
        channel_revenue = df.groupby('Sales_Channel')['Total_Revenue'].sum()
        top_channel = channel_revenue.idxmax()
        top_channel_pct = (channel_revenue.max() / channel_revenue.sum()) * 100
        insights.append(f"<p>🏆 <strong>Top Channel:</strong> {top_channel} ({top_channel_pct:.1f}% of revenue)</p>")
    
    return "".join(insights) if insights else "<p>Analyzing revenue patterns...</p>"

def generate_operational_insights(df):
    """Generate operational insights"""
    insights = []
    
    if len(df) == 0:
        return "<p>No operational data available.</p>"
    
    # Lead time analysis
    if 'Total_Lead_Time' in df.columns:
        avg_lead_time = df['Total_Lead_Time'].mean()
        median_lead_time = df['Total_Lead_Time'].median()
        std_lead_time = df['Total_Lead_Time'].std()
        
        insights.append(f"<p>⏱️ <strong>Avg Lead Time:</strong> {avg_lead_time:.1f} days</p>")
        
        if std_lead_time > avg_lead_time * 0.3:
            insights.append("<p>⚠️ High lead time variability detected - consider process standardization</p>")
        
        # Lead time by channel
        if 'Sales_Channel' in df.columns:
            channel_lead_times = df.groupby('Sales_Channel')['Total_Lead_Time'].mean()
            fastest_channel = channel_lead_times.idxmin()
            slowest_channel = channel_lead_times.idxmax()
            insights.append(f"<p>🏃 <strong>Fastest:</strong> {fastest_channel} ({channel_lead_times.min():.1f} days)</p>")
            insights.append(f"<p>🐌 <strong>Slowest:</strong> {slowest_channel} ({channel_lead_times.max():.1f} days)</p>")
    
    # Warehouse efficiency
    if 'WarehouseCode' in df.columns:
        warehouse_orders = df['WarehouseCode'].value_counts()
        top_warehouse = warehouse_orders.index[0]
        insights.append(f"<p>🏭 <strong>Busiest Warehouse:</strong> {top_warehouse} ({warehouse_orders.iloc[0]} orders)</p>")
    
    return "".join(insights) if insights else "<p>Analyzing operational metrics...</p>"

def generate_performance_insights(df):
    """Generate performance insights"""
    insights = []
    
    if len(df) == 0:
        return "<p>No performance data available.</p>"
    
    # Profit analysis
    if 'Profit' in df.columns and 'Total_Revenue' in df.columns:
        total_profit = df['Profit'].sum()
        total_revenue = df['Total_Revenue'].sum()
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        insights.append(f"<p>💎 <strong>Total Profit:</strong> ${total_profit:,.0f}</p>")
        insights.append(f"<p>📊 <strong>Profit Margin:</strong> {profit_margin:.1f}%</p>")
        
        if profit_margin > 20:
            insights.append("<p>✅ Strong profitability performance</p>")
        elif profit_margin < 5:
            insights.append("<p>⚠️ Low profit margins - review pricing strategy</p>")
    
    # Order volume analysis
    total_orders = len(df)
    insights.append(f"<p>📦 <strong>Total Orders:</strong> {total_orders:,}</p>")
    
    # Peak performance analysis
    if 'OrderDate' in df.columns:
        daily_orders = df.groupby(df['OrderDate'].dt.date).size()
        peak_day = daily_orders.idxmax()
        peak_orders = daily_orders.max()
        insights.append(f"<p>🎯 <strong>Peak Day:</strong> {peak_day} ({peak_orders} orders)</p>")
    
    # Customer distribution
    if 'CustomerID' in df.columns:
        unique_customers = df['CustomerID'].nunique()
        orders_per_customer = total_orders / unique_customers if unique_customers > 0 else 0
        insights.append(f"<p>👥 <strong>Unique Customers:</strong> {unique_customers:,}</p>")
        insights.append(f"<p>🔄 <strong>Orders per Customer:</strong> {orders_per_customer:.1f}</p>")
    
    return "".join(insights) if insights else "<p>Analyzing performance metrics...</p>"


def show_interactive_dashboard(analyzer, llm_assistant):
    """Enhanced interactive dashboard with quality insights and visualizations"""

    st.markdown(
        '<div class="main-header">📊 Advanced Supply Chain Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    if analyzer.processed_df is None or analyzer.processed_df.empty:
        st.error("No data loaded. Please check the data file path.")
        return

    # Interactive filters
    st.markdown(
        '<div class="section-header">🎛️ Interactive Data Filters</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if "OrderDate" in analyzer.processed_df.columns:
            min_date = analyzer.processed_df["OrderDate"].min().date()
            max_date = analyzer.processed_df["OrderDate"].max().date()
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

    with col2:
        if "Sales_Channel" in analyzer.processed_df.columns:
            channels = st.multiselect(
                "Select Sales Channels",
                options=analyzer.processed_df["Sales_Channel"].unique(),
                default=analyzer.processed_df["Sales_Channel"].unique(),
            )

    with col3:
        if "WarehouseCode" in analyzer.processed_df.columns:
            warehouses = st.multiselect(
                "Select Warehouses",
                options=analyzer.processed_df["WarehouseCode"].unique(),
                default=analyzer.processed_df["WarehouseCode"].unique(),
            )

    with col4:
        aggregation_level = st.selectbox(
            "Time Aggregation", options=["daily", "weekly", "monthly"], index=2
        )

    # Get filtered time series data
    time_series_data = analyzer.create_advanced_time_series_data(
        date_range=date_range if "date_range" in locals() else None,
        channels=channels if "channels" in locals() else None,
        warehouses=warehouses if "warehouses" in locals() else None,
    )

    # Calculate filtered metrics
    filtered_df = analyzer.processed_df.copy()
    if "date_range" in locals() and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df["OrderDate"] >= start_date)
            & (filtered_df["OrderDate"] <= end_date)
        ]

    if "channels" in locals() and channels:
        filtered_df = filtered_df[filtered_df["Sales_Channel"].isin(channels)]

    if "warehouses" in locals() and warehouses:
        filtered_df = filtered_df[filtered_df["WarehouseCode"].isin(warehouses)]

    # Enhanced KPI metrics with better calculations
    st.markdown(
        '<div class="section-header">📈 Dynamic Key Performance Indicators</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_revenue = (
            filtered_df["Total_Revenue"].sum()
            if "Total_Revenue" in filtered_df.columns and len(filtered_df) > 0
            else 0
        )
        # Calculate growth rate if possible
        growth_rate = 0
        if aggregation_level in time_series_data and len(time_series_data[aggregation_level]) > 1:
            ts_data = time_series_data[aggregation_level]
            if len(ts_data) >= 2:
                recent_avg = ts_data["Total_Revenue"].tail(3).mean()
                older_avg = ts_data["Total_Revenue"].head(3).mean()
                if older_avg > 0:
                    growth_rate = ((recent_avg - older_avg) / older_avg) * 100

        growth_indicator = "📈" if growth_rate > 0 else "📉" if growth_rate < 0 else "➡️"
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>💰 Total Revenue</h4>
            <h2>${total_revenue:,.0f}</h2>
            <small>{growth_indicator} {growth_rate:+.1f}% trend</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        total_orders = len(filtered_df)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>📦 Total Orders</h4>
            <h2>{total_orders:,}</h2>
            <small>AOV: ${avg_order_value:,.0f}</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        if "Profit" in filtered_df.columns and len(filtered_df) > 0:
            total_profit = filtered_df["Profit"].sum()
            profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>💎 Total Profit</h4>
                <h2>${total_profit:,.0f}</h2>
                <small>Margin: {profit_margin:.1f}%</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col4:
        if "Total_Lead_Time" in filtered_df.columns and len(filtered_df) > 0:
            avg_lead_time = filtered_df["Total_Lead_Time"].mean()
            median_lead_time = filtered_df["Total_Lead_Time"].median()
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>⏱️ Avg Lead Time</h4>
                <h2>{avg_lead_time:.1f}</h2>
                <small>Median: {median_lead_time:.1f} days</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col5:
        top_performing_channel = "N/A"
        if "Sales_Channel" in filtered_df.columns and len(filtered_df) > 0:
            channel_performance = filtered_df.groupby("Sales_Channel")["Total_Revenue"].sum()
            top_performing_channel = channel_performance.idxmax() if len(channel_performance) > 0 else "N/A"
            top_channel_revenue = channel_performance.max() if len(channel_performance) > 0 else 0
            
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>🏆 Top Channel</h4>
                <h2>{top_performing_channel}</h2>
                <small>${top_channel_revenue:,.0f}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Advanced visualizations section
    st.markdown(
        '<div class="section-header">📊 Advanced Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Row 1: Revenue Trend and Sales Channel Performance
    col1, col2 = st.columns([3, 2])

    with col1:
        # Visualization 1: Enhanced Revenue Trend with Moving Average
        if (
            aggregation_level in time_series_data
            and not time_series_data[aggregation_level].empty
        ):
            data = time_series_data[aggregation_level]
            time_col = (
                "OrderDate"
                if aggregation_level == "daily"
                else ("Week" if aggregation_level == "weekly" else "Month")
            )

            # Calculate moving average
            data = data.copy()
            window_size = min(3, len(data) // 3) if len(data) > 3 else 1
            data['Revenue_MA'] = data['Total_Revenue'].rolling(window=window_size, center=True).mean()

            fig = go.Figure()
            
            # Add revenue line
            fig.add_trace(go.Scatter(
                x=data[time_col],
                y=data['Total_Revenue'],
                name='Revenue',
                line=dict(width=3, color='#667eea'),
                mode='lines+markers',
                marker=dict(size=6)
            ))
            
            # Add moving average
            fig.add_trace(go.Scatter(
                x=data[time_col],
                y=data['Revenue_MA'],
                name=f'Moving Avg ({window_size})',
                line=dict(width=2, color='#ff6b6b', dash='dash'),
                mode='lines'
            ))

            fig.update_layout(
                title=f"📈 Revenue Trend Analysis - {aggregation_level.title()}",
                template="plotly_white",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                title_font_size=16,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Visualization 2: Sales Channel Performance
        if "Sales_Channel" in filtered_df.columns and len(filtered_df) > 0:
            channel_metrics = filtered_df.groupby("Sales_Channel").agg({
                'Total_Revenue': 'sum',
                'OrderNumber': 'count',
                'Profit': 'sum'
            }).reset_index()
            
            fig = px.pie(
                channel_metrics,
                values='Total_Revenue',
                names='Sales_Channel',
                title='💼 Revenue by Sales Channel',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=400,
                template="plotly_white",
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)

    # Row 2: Warehouse Performance and Order Quantity Distribution
    col1, col2 = st.columns(2)

    with col1:
        # Visualization 3: Warehouse Performance Heatmap
        if "WarehouseCode" in filtered_df.columns and len(filtered_df) > 0:
            # Create warehouse performance metrics
            warehouse_perf = filtered_df.groupby("WarehouseCode").agg({
                'Total_Revenue': 'sum',
                'OrderNumber': 'count',
                'Total_Lead_Time': 'mean',
                'Profit': 'sum'
            }).reset_index()
            
            fig = px.bar(
                warehouse_perf.head(10),  # Top 10 warehouses
                x='WarehouseCode',
                y='Total_Revenue',
                color='Total_Lead_Time',
                title='🏭 Warehouse Performance Analysis',
                color_continuous_scale='RdYlBu_r',
                labels={'Total_Lead_Time': 'Avg Lead Time (Days)'}
            )
            fig.update_layout(
                height=400,
                template="plotly_white",
                font=dict(size=12),
                title_font_size=16,
                xaxis={'categoryorder': 'total descending'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Visualization 4: Order Value Distribution Analysis
        if "Total_Revenue" in filtered_df.columns and len(filtered_df) > 0:
            # Create order value segments
            filtered_df_copy = filtered_df.copy()
            filtered_df_copy['Order_Value_Segment'] = pd.cut(
                filtered_df_copy['Total_Revenue'],
                bins=[0, 100, 500, 1000, 5000, float('inf')],
                labels=['<$100', '$100-500', '$500-1K', '$1K-5K', '>$5K']
            )
            
            segment_counts = filtered_df_copy['Order_Value_Segment'].value_counts()
            
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                title='💰 Order Value Distribution',
                color=segment_counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                height=400,
                template="plotly_white",
                font=dict(size=12),
                title_font_size=16,
                xaxis_title="Order Value Segment",
                yaxis_title="Number of Orders",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    # Row 3: Time-based Analysis and Performance Metrics
    col1, col2 = st.columns(2)

    with col1:
        # Visualization 5: Daily/Weekly Pattern Analysis
        if len(filtered_df) > 0 and "OrderDate" in filtered_df.columns:
            # Analyze patterns by day of week or month
            if aggregation_level == "daily":
                pattern_df = filtered_df.copy()
                pattern_df['DayOfWeek'] = pattern_df['OrderDate'].dt.day_name()
                pattern_data = pattern_df.groupby('DayOfWeek')['Total_Revenue'].mean().reset_index()
                
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                pattern_data['DayOfWeek'] = pd.Categorical(pattern_data['DayOfWeek'], categories=day_order, ordered=True)
                pattern_data = pattern_data.sort_values('DayOfWeek')
                
                fig = px.bar(
                    pattern_data,
                    x='DayOfWeek',
                    y='Total_Revenue',
                    title='📅 Average Daily Revenue Patterns',
                    color='Total_Revenue',
                    color_continuous_scale='blues'
                )
            else:
                pattern_df = filtered_df.copy()
                pattern_df['Month'] = pattern_df['OrderDate'].dt.month_name()
                pattern_data = pattern_df.groupby('Month')['Total_Revenue'].sum().reset_index()
                
                fig = px.line(
                    pattern_data,
                    x='Month',
                    y='Total_Revenue',
                    title='📈 Monthly Revenue Patterns',
                    markers=True
                )
                fig.update_traces(line=dict(width=3, color='#667eea'), marker=dict(size=8))
            
            fig.update_layout(
                height=400,
                template="plotly_white",
                font=dict(size=12),
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Visualization 6: Lead Time vs Order Value Correlation
        if all(col in filtered_df.columns for col in ['Total_Lead_Time', 'Total_Revenue']) and len(filtered_df) > 0:
            # Remove outliers for better visualization
            q1_revenue = filtered_df['Total_Revenue'].quantile(0.25)
            q3_revenue = filtered_df['Total_Revenue'].quantile(0.75)
            q1_lead = filtered_df['Total_Lead_Time'].quantile(0.25)
            q3_lead = filtered_df['Total_Lead_Time'].quantile(0.75)
            
            clean_df = filtered_df[
                (filtered_df['Total_Revenue'] >= q1_revenue) & 
                (filtered_df['Total_Revenue'] <= q3_revenue * 1.5) &
                (filtered_df['Total_Lead_Time'] >= q1_lead) & 
                (filtered_df['Total_Lead_Time'] <= q3_lead * 1.5) &
                (filtered_df['Total_Lead_Time'].notna())
            ]
            
            if len(clean_df) > 0:
                fig = px.scatter(
                    clean_df.sample(min(1000, len(clean_df))),  # Sample for performance
                    x='Total_Lead_Time',
                    y='Total_Revenue',
                    color='Sales_Channel' if 'Sales_Channel' in clean_df.columns else None,
                    title='⏱️ Lead Time vs Order Value Analysis',
                    opacity=0.6,
                    size_max=10
                )
                
                # Add trend line
                from sklearn.linear_model import LinearRegression
                import numpy as np
                
                X = clean_df[['Total_Lead_Time']].dropna()
                y = clean_df.loc[X.index, 'Total_Revenue']
                
                if len(X) > 10:
                    reg = LinearRegression().fit(X, y)
                    trend_x = np.linspace(X.min(), X.max(), 100)
                    trend_y = reg.predict(trend_x.reshape(-1, 1))
                    
                    fig.add_trace(go.Scatter(
                        x=trend_x.flatten(),
                        y=trend_y,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    height=400,
                    template="plotly_white",
                    font=dict(size=12),
                    title_font_size=16
                )
                st.plotly_chart(fig, use_container_width=True)

    # AI-powered insights section
    st.markdown(
        '<div class="section-header">🤖 AI-Powered Business Insights</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # Generate revenue insights
        if len(filtered_df) > 0:
            revenue_insights = generate_revenue_insights(filtered_df, time_series_data.get(aggregation_level))
            st.markdown(
                f'<div class="insight-box"><h4>💰 Revenue Analysis</h4>{revenue_insights}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        # Generate operational insights
        if len(filtered_df) > 0:
            operational_insights = generate_operational_insights(filtered_df)
            st.markdown(
                f'<div class="insight-box"><h4>⚙️ Operational Insights</h4>{operational_insights}</div>',
                unsafe_allow_html=True,
            )

    with col3:
        # Generate performance insights
        if len(filtered_df) > 0:
            performance_insights = generate_performance_insights(filtered_df)
            st.markdown(
                f'<div class="insight-box"><h4>📊 Performance Metrics</h4>{performance_insights}</div>',
                unsafe_allow_html=True,
            )


def show_forecasting_page(analyzer, llm_assistant):
    """Advanced time series forecasting page"""

    st.markdown(
        '<div class="main-header">🔮 Advanced Time Series Forecasting</div>',
        unsafe_allow_html=True,
    )

    if analyzer.processed_df is None or analyzer.processed_df.empty:
        st.error("No data loaded for forecasting.")
        return

    # Forecasting controls
    st.markdown(
        '<div class="section-header">⚙️ Forecasting Configuration</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        forecast_periods = st.slider("Forecast Periods", 3, 24, 12)

    with col2:
        target_metric = st.selectbox(
            "Target Metric", ["Total_Revenue", "Order_Quantity", "Profit"]
        )

    with col3:
        aggregation = st.selectbox("Time Aggregation", ["monthly", "weekly"], index=0)

    with col4:
        if st.button("🚀 Generate Forecasts", type="primary"):
            st.session_state.run_forecast = True

    # Get time series data
    time_series_data = analyzer.create_advanced_time_series_data()

    if aggregation in time_series_data and not time_series_data[aggregation].empty:
        data = time_series_data[aggregation]

        if "run_forecast" in st.session_state and st.session_state.run_forecast:
            # Generate multiple forecasts
            with st.spinner("🔬 Running advanced forecasting models..."):
                arima_result = analyzer.advanced_arima_forecast(
                    data, target_metric, forecast_periods
                )
                sarima_result = analyzer.sarima_forecast(
                    data, target_metric, forecast_periods
                )
                holt_result = analyzer.triple_exponential_smoothing(
                    data, target_metric, forecast_periods
                )

            # Forecast visualization
            st.markdown(
                '<div class="section-header">📊 Forecast Results Comparison</div>',
                unsafe_allow_html=True,
            )

            fig = go.Figure()

            # Historical data
            time_col = "Week" if aggregation == "weekly" else "Month"
            fig.add_trace(
                go.Scatter(
                    x=data[time_col],
                    y=data[target_metric],
                    name="Historical Data",
                    line=dict(width=4, color="#1f77b4"),
                )
            )

            # Forecast periods
            last_period = data[time_col].iloc[-1]
            if aggregation == "monthly":
                forecast_periods_labels = [f"F{i + 1}" for i in range(forecast_periods)]
            else:
                forecast_periods_labels = [f"W{i + 1}" for i in range(forecast_periods)]

            # ARIMA forecast
            if arima_result and "forecast" in arima_result:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_periods_labels,
                        y=arima_result["forecast"],
                        name=f'ARIMA{arima_result.get("parameters", "")}',
                        line=dict(width=3, color="#ff6b6b", dash="dash"),
                    )
                )

                if "upper_ci" in arima_result and "lower_ci" in arima_result:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_periods_labels,
                            y=arima_result["upper_ci"],
                            fill=None,
                            mode="lines",
                            line_color="rgba(255,107,107,0)",
                            showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_periods_labels,
                            y=arima_result["lower_ci"],
                            fill="tonexty",
                            mode="lines",
                            line_color="rgba(255,107,107,0)",
                            name="ARIMA Confidence Interval",
                            fillcolor="rgba(255,107,107,0.2)",
                        )
                    )

            # SARIMA forecast
            if sarima_result and "forecast" in sarima_result:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_periods_labels,
                        y=sarima_result["forecast"],
                        name="SARIMA",
                        line=dict(width=3, color="#2ecc71", dash="dash"),
                    )
                )

            # Holt-Winters forecast
            if holt_result and "forecast" in holt_result:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_periods_labels,
                        y=holt_result["forecast"],
                        name="Holt-Winters",
                        line=dict(width=3, color="#f39c12", dash="dash"),
                    )
                )

            fig.update_layout(
                title=f"🔮 {target_metric} Forecasting - Multiple Models Comparison",
                template="plotly_white",
                height=600,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
            )

            st.plotly_chart(fig, use_container_width=True)

            # Model performance and insights
            st.markdown(
                '<div class="section-header">📊 Model Performance & Statistical Analysis</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if arima_result is not None:
                    aic_value = arima_result.get("aic", "N/A")
                    aic_display = (
                        f"{aic_value:.2f}"
                        if isinstance(aic_value, (int, float))
                        else "N/A"
                    )
                    
                    stationarity = arima_result.get("stationarity", {})
                    is_stationary = stationarity.get("is_stationary", False) if stationarity else False

                    st.markdown(
                        f"""
                        <div class="model-performance">
                            <h4>🔄 ARIMA Model</h4>
                            <p><strong>Parameters:</strong> {arima_result.get("parameters", "N/A")}</p>
                            <p><strong>AIC:</strong> {aic_display}</p>
                            <p><strong>Stationarity:</strong> {"✅" if is_stationary else "❌"}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="model-performance">
                            <h4>🔄 ARIMA Model</h4>
                            <p><strong>Status:</strong> ❌ Model failed</p>
                            <p><strong>Error:</strong> Unable to fit ARIMA model</p>
                            <p><strong>Fallback:</strong> Using simple forecast</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col2:
                if sarima_result is not None:
                    aic_value = sarima_result.get("aic", "N/A")
                    aic_display = (
                        f"{aic_value:.2f}"
                        if isinstance(aic_value, (int, float))
                        else "N/A"
                    )

                    st.markdown(
                        f"""
                        <div class="model-performance">
                            <h4>📈 SARIMA Model</h4>
                            <p><strong>Seasonal Periods:</strong> {sarima_result.get("seasonal_periods", "N/A")}</p>
                            <p><strong>AIC:</strong> {aic_display}</p>
                            <p><strong>Seasonality:</strong> {"✅" if sarima_result.get("seasonal_periods", 0) > 0 else "❌"}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="model-performance">
                            <h4>📈 SARIMA Model</h4>
                            <p><strong>Status:</strong> ❌ Model failed</p>
                            <p><strong>Error:</strong> Unable to fit SARIMA model</p>
                            <p><strong>Fallback:</strong> Using simple forecast</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col3:
                if holt_result is not None:
                    st.markdown(
                        f"""
                        <div class="model-performance">
                            <h4>🌊 Holt-Winters</h4>
                            <p><strong>Seasonal Periods:</strong> {holt_result.get("seasonal_periods", "N/A")}</p>
                            <p><strong>Method:</strong> Triple Exponential</p>
                            <p><strong>Components:</strong> Trend + Seasonal</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="model-performance">
                            <h4>🌊 Holt-Winters</h4>
                            <p><strong>Status:</strong> ❌ Model failed</p>
                            <p><strong>Error:</strong> Unable to fit Holt-Winters model</p>
                            <p><strong>Fallback:</strong> Using simple forecast</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # AI-generated forecast analysis
            st.markdown(
                '<div class="section-header">🤖 AI Forecast Analysis</div>',
                unsafe_allow_html=True,
            )

            # Safely get forecast values
            arima_forecast_avg = (
                np.mean(arima_result.get("forecast", [0])) if arima_result else 0
            )
            sarima_forecast_avg = (
                np.mean(sarima_result.get("forecast", [0])) if sarima_result else 0
            )
            holt_forecast_avg = (
                np.mean(holt_result.get("forecast", [0])) if holt_result else 0
            )

            forecast_context = f"""
            Forecasting Analysis for {target_metric}:
            - Historical data points: {len(data)}
            - Forecast periods: {forecast_periods}
            - ARIMA forecast average: ${arima_forecast_avg:,.0f}
            - SARIMA forecast average: ${sarima_forecast_avg:,.0f}
            - Holt-Winters forecast average: ${holt_forecast_avg:,.0f}
            - Best ARIMA parameters: {arima_result.get("parameters", "N/A") if arima_result else "N/A"}
            - Stationarity test p-value: {(arima_result.get("stationarity") or {}).get("p_value", "N/A") if arima_result else "N/A"}
            """

            forecast_analysis = llm_assistant.get_structured_response(
                "Analyze these time series forecasting results and provide insights on model performance, forecast reliability, and business recommendations.",
                forecast_context,
            )

            st.markdown(
                f'<div class="ai-response">{forecast_analysis}</div>',
                unsafe_allow_html=True,
            )


def show_enhanced_research_assistant(analyzer, llm_assistant):
    """Enhanced AI research assistant with advanced capabilities"""

    st.markdown(
        '<div class="main-header">🤖 Advanced AI Research Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="insight-box">
    <h3>🎯 Advanced Analytics Capabilities</h3>
    <p>This AI assistant combines statistical analysis, time series modeling, and business intelligence to provide
    comprehensive insights into your supply chain data. It can perform advanced forecasting, identify patterns,
    and provide actionable recommendations.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Enhanced context generation
    context = generate_comprehensive_context(analyzer)

    # Advanced research questions
    st.markdown("### 💡 Advanced Research Questions")

    advanced_questions = [
        "Perform comprehensive time series decomposition and identify seasonal patterns with statistical significance",
        "Compare ARIMA vs SARIMA model performance for inventory optimization and provide implementation roadmap",
        "Analyze supply chain variability using statistical process control and recommend improvement strategies",
        "Evaluate the impact of different sales channels on forecasting accuracy and profitability",
        "Conduct advanced correlation analysis between lead times, order patterns, and customer satisfaction metrics",
        "Design an optimal inventory management strategy using time series forecasting and economic order quantity principles",
    ]

    col1, col2 = st.columns(2)

    for i, question in enumerate(advanced_questions):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"🎯 {question[:50]}...", key=f"adv_q_{i}"):
                st.session_state.research_question = question

    # Research question input
    st.markdown("### 🔍 Ask an Advanced Research Question")
    research_question = st.text_area(
        "Enter your detailed research question:",
        value=st.session_state.get("research_question", ""),
        height=120,
        placeholder="e.g., Using advanced time series analysis, evaluate the seasonal impact on inventory levels and develop a predictive model for optimal stock management...",
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("🚀 Generate Advanced AI Analysis", type="primary"):
            if research_question:
                with st.spinner(
                    "🔬 Performing comprehensive analysis with advanced AI models..."
                ):
                    # Generate additional analysis data
                    analysis_data = generate_analysis_data(analyzer)

                    response = llm_assistant.get_structured_response(
                        research_question, context, analysis_data
                    )

                    st.markdown("### 📋 Comprehensive AI Analysis")
                    st.markdown(
                        f'<div class="ai-response">{response}</div>',
                        unsafe_allow_html=True,
                    )

    with col2:
        st.markdown("### 🔧 Analysis Tools")
        if st.button("📊 Generate Data Summary"):
            summary = generate_data_summary(analyzer)
            st.markdown(
                f'<div class="insight-box">{summary}</div>', unsafe_allow_html=True
            )


def show_model_comparison(analyzer, llm_assistant):
    """Model comparison and evaluation page"""

    st.markdown(
        '<div class="main-header">📈 Advanced Model Comparison & Evaluation</div>',
        unsafe_allow_html=True,
    )

    if analyzer.processed_df is None or analyzer.processed_df.empty:
        st.error("No data loaded for model comparison.")
        return

    st.markdown(
        """
    <div class="insight-box">
    <h3>🎯 Model Evaluation Framework</h3>
    <p>Compare different time series forecasting models using comprehensive statistical metrics
    and cross-validation techniques to determine the best approach for your supply chain data.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Model comparison interface would go here
    st.markdown("### 🔄 Model Comparison Results")
    st.info(
        "This section will compare multiple forecasting models and provide detailed performance metrics."
    )


def generate_comprehensive_context(analyzer):
    """Generate comprehensive context for AI analysis"""
    if analyzer.processed_df is None:
        return "No data available for analysis."

    df = analyzer.processed_df
    
    # Safe attribute access with default values
    total_records = len(df)
    date_range_str = "N/A"
    total_revenue = 0
    avg_order_value = 0
    total_profit = 0
    avg_profit_margin = 0
    avg_lead_time = "N/A"
    sales_channels = 0
    warehouses = 0

    try:
        if "OrderDate" in df.columns and not df["OrderDate"].empty:
            min_date = df["OrderDate"].min()
            max_date = df["OrderDate"].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    except:
        pass

    try:
        if "Total_Revenue" in df.columns:
            total_revenue = df["Total_Revenue"].sum()
            avg_order_value = df["Total_Revenue"].mean()
    except:
        pass

    try:
        if "Profit" in df.columns:
            total_profit = df["Profit"].sum()
    except:
        pass

    try:
        if "Profit_Margin" in df.columns:
            avg_profit_margin = df["Profit_Margin"].mean()
    except:
        pass

    try:
        if "Total_Lead_Time" in df.columns:
            avg_lead_time = f"{df['Total_Lead_Time'].mean():.1f} days"
    except:
        pass

    try:
        if "Sales_Channel" in df.columns:
            sales_channels = df["Sales_Channel"].nunique()
    except:
        pass

    try:
        if "WarehouseCode" in df.columns:
            warehouses = df["WarehouseCode"].nunique()
    except:
        pass

    context = f"""
    ## Comprehensive Supply Chain Dataset Analysis

    ### Dataset Overview:
    - Total Records: {total_records:,}
    - Date Range: {date_range_str}

    ### Financial Metrics:
    - Total Revenue: ${total_revenue:,.2f}
    - Average Order Value: ${avg_order_value:,.2f}
    - Total Profit: ${total_profit:,.2f}
    - Average Profit Margin: {avg_profit_margin:.2f}%

    ### Operational Metrics:
    - Average Lead Time: {avg_lead_time}
    - Sales Channels: {sales_channels}
    - Warehouses: {warehouses}
    """

    return context


def generate_analysis_data(analyzer):
    """Generate additional analysis data for AI context"""
    if analyzer.processed_df is None:
        return "No analysis data available."

    try:
        time_series_data = analyzer.create_advanced_time_series_data()
        if "monthly" in time_series_data and not time_series_data["monthly"].empty:
            monthly_data = time_series_data["monthly"]

            revenue_trend = "Stable"
            try:
                if len(monthly_data) >= 2:
                    if monthly_data["Total_Revenue"].iloc[-1] > monthly_data["Total_Revenue"].iloc[0]:
                        revenue_trend = "Upward"
                    else:
                        revenue_trend = "Downward"
            except:
                pass

            volatility = 0
            try:
                mean_rev = monthly_data["Total_Revenue"].mean()
                if mean_rev > 0:
                    volatility = monthly_data["Total_Revenue"].std() / mean_rev
            except:
                pass

            analysis = f"""
            Time Series Analysis Results:
            - Monthly data points: {len(monthly_data)}
            - Revenue trend: {revenue_trend}
            - Revenue volatility (CV): {volatility:.2f}
            - Seasonal patterns detected: {len(monthly_data) >= 12}
            """
            return analysis
    except Exception as e:
        return f"Error in analysis: {str(e)}"

    return "Basic analysis data available."


def generate_data_summary(analyzer):
    """Generate a comprehensive data summary"""
    if analyzer.processed_df is None:
        return "No data available for summary."

    df = analyzer.processed_df

    # Safe calculations with default values
    total_records = len(df)
    date_coverage = "N/A"
    missing_values = df.isnull().sum().sum()
    unique_customers = "N/A"
    unique_products = "N/A"
    revenue_mean = 0
    revenue_std = 0

    try:
        if "OrderDate" in df.columns and not df["OrderDate"].empty:
            date_range = df["OrderDate"].max() - df["OrderDate"].min()
            date_coverage = f"{date_range.days} days"
    except:
        pass

    try:
        if "CustomerID" in df.columns:
            unique_customers = df["CustomerID"].nunique()
    except:
        pass

    try:
        if "ProductID" in df.columns:
            unique_products = df["ProductID"].nunique()
    except:
        pass

    try:
        if "Total_Revenue" in df.columns:
            revenue_mean = df["Total_Revenue"].mean()
            revenue_std = df["Total_Revenue"].std()
    except:
        pass

    summary = f"""
    <h4>📊 Comprehensive Data Summary</h4>
    <ul>
        <li><strong>Total Records:</strong> {total_records:,}</li>
        <li><strong>Date Coverage:</strong> {date_coverage}</li>
        <li><strong>Missing Values:</strong> {missing_values} total</li>
        <li><strong>Unique Customers:</strong> {unique_customers}</li>
        <li><strong>Unique Products:</strong> {unique_products}</li>
        <li><strong>Revenue Distribution:</strong> Mean ${revenue_mean:,.0f}, Std ${revenue_std:,.0f}</li>
    </ul>
    """

    return summary


if __name__ == "__main__":
    # Initialize session state
    if "research_question" not in st.session_state:
        st.session_state.research_question = ""
    if "run_forecast" not in st.session_state:
        st.session_state.run_forecast = False

    main()