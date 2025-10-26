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
warnings.filterwarnings('ignore')

# Time series forecasting imports
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

# LLM imports
import os
from dotenv import load_dotenv
import groq

# Page configuration
st.set_page_config(
    page_title="Supply Chain Analytics & AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 4px solid #ff6b6b;
        padding-left: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .forecast-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class SupplyChainAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the supply chain data"""
        try:
            # Read the CSV file - skip the problematic row with dollar amounts
            self.df = pd.read_csv(self.data_path)
            
            # Clean the data - remove rows that are not actual data records
            self.df = self.df[~self.df.iloc[:, 0].str.contains('\$', na=False)]
            
            # Reset column names based on the actual data structure
            new_columns = [
                'OrderNumber', 'Sales_Channel', 'Warehouse', 'Order_Date', 
                'Due_Date', 'Ship_Date', 'Delivery_Date', 'Currency',
                'Quantity', 'Unit_Price', 'Product_ID', 'Customer_ID',
                'Shipping_Days', 'Discount_Rate', 'Shipping_Cost', 'Total_Revenue'
            ]
            
            # Assign new column names if we have the right number of columns
            if len(self.df.columns) >= len(new_columns):
                self.df = self.df.iloc[:, :len(new_columns)]
                self.df.columns = new_columns[:len(self.df.columns)]
            
            # Basic data cleaning
            self.clean_data()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def clean_data(self):
        """Clean and preprocess the data"""
        try:
            # Convert date columns to datetime
            date_columns = ['Order_Date', 'Due_Date', 'Ship_Date', 'Delivery_Date']
            for col in date_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce', dayfirst=True)
            
            # Convert numeric columns - handle currency symbols and commas
            numeric_columns = ['Quantity', 'Unit_Price', 'Total_Revenue', 'Shipping_Cost', 'Discount_Rate']
            for col in numeric_columns:
                if col in self.df.columns:
                    # Remove dollar signs and commas, then convert to numeric
                    self.df[col] = self.df[col].astype(str).str.replace('$', '').str.replace(',', '')
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Create additional time-based features
            if all(col in self.df.columns for col in ['Order_Date', 'Ship_Date', 'Delivery_Date']):
                self.df['Order_to_Ship_Days'] = (self.df['Ship_Date'] - self.df['Order_Date']).dt.days
                self.df['Ship_to_Delivery_Days'] = (self.df['Delivery_Date'] - self.df['Ship_Date']).dt.days
                self.df['Total_Lead_Time'] = (self.df['Delivery_Date'] - self.df['Order_Date']).dt.days
            
            # Remove any rows with invalid Order_Date (essential for time series)
            if 'Order_Date' in self.df.columns:
                self.df = self.df[self.df['Order_Date'].notna()]
            
        except Exception as e:
            st.error(f"Error cleaning data: {e}")
    
    def create_time_series_analysis(self):
        """Create comprehensive time series analysis"""
        analysis = {}
        
        try:
            # Monthly trends - convert to string immediately to avoid Period serialization issues
            if 'Order_Date' in self.df.columns:
                self.df['Order_Month'] = self.df['Order_Date'].dt.to_period('M').astype(str)
                monthly_data = self.df.groupby('Order_Month').agg({
                    'Total_Revenue': 'sum',
                    'Quantity': 'sum',
                    'OrderNumber': 'count'
                }).reset_index()
                analysis['monthly_trends'] = monthly_data
            
            # Weekly trends for more granular analysis
            if 'Order_Date' in self.df.columns:
                self.df['Order_Week'] = self.df['Order_Date'].dt.to_period('W').astype(str)
                weekly_data = self.df.groupby('Order_Week').agg({
                    'Total_Revenue': 'sum',
                    'Quantity': 'sum',
                    'OrderNumber': 'count'
                }).reset_index()
                analysis['weekly_trends'] = weekly_data
            
            # Channel performance over time
            if 'Sales_Channel' in self.df.columns and 'Order_Month' in self.df.columns:
                channel_trends = self.df.groupby(['Order_Month', 'Sales_Channel']).agg({
                    'Total_Revenue': 'sum',
                    'OrderNumber': 'count'
                }).reset_index()
                analysis['channel_trends'] = channel_trends
            
            # Warehouse performance
            if 'Warehouse' in self.df.columns and 'Order_Month' in self.df.columns:
                warehouse_trends = self.df.groupby(['Order_Month', 'Warehouse']).agg({
                    'Total_Revenue': 'sum',
                    'OrderNumber': 'count'
                }).reset_index()
                analysis['warehouse_trends'] = warehouse_trends
                
        except Exception as e:
            st.error(f"Error in time series analysis: {e}")
        
        return analysis
    
    def arima_forecast(self, data, column='Total_Revenue', periods=6):
        """ARIMA forecasting using statsmodels"""
        try:
            values = data[column].values
            
            # Fit ARIMA model
            model = ARIMA(values, order=(2, 1, 2))  # (p,d,q) parameters
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=periods)
            return forecast.tolist()
            
        except Exception as e:
            st.warning(f"ARIMA forecast failed: {e}")
            # Fallback to simple moving average
            return self.simple_forecast(data, column, periods)
    
    def holt_winters_forecast(self, data, column='Total_Revenue', periods=6):
        """Holt-Winters exponential smoothing forecast"""
        try:
            values = data[column].values
            
            # Fit Holt-Winters model
            model = ExponentialSmoothing(values, trend='add', seasonal='add', seasonal_periods=4)
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(periods)
            return forecast.tolist()
            
        except Exception as e:
            st.warning(f"Holt-Winters forecast failed: {e}")
            return self.simple_forecast(data, column, periods)
    
    def prophet_style_forecast(self, data, column='Total_Revenue', periods=6):
        """Custom forecasting method inspired by Facebook Prophet"""
        try:
            values = data[column].values
            
            # Calculate trend component using linear regression
            x = np.arange(len(values))
            trend_coef = np.polyfit(x, values, 1)[0]
            
            # Calculate seasonal component (simple approach)
            if len(values) >= 8:  # Need enough data for seasonality
                seasonal = np.array([values[i] - np.mean(values) for i in range(len(values))])
                seasonal_pattern = np.tile(seasonal[-4:], periods // 4 + 1)[:periods]
            else:
                seasonal_pattern = np.zeros(periods)
            
            # Generate forecast with trend and seasonality
            last_value = values[-1]
            forecast = []
            for i in range(periods):
                next_val = last_value + trend_coef * (i + 1) + seasonal_pattern[i] * 0.3
                forecast.append(max(next_val, 0))  # Ensure non-negative
            
            return forecast
            
        except Exception as e:
            st.warning(f"Prophet-style forecast failed: {e}")
            return self.simple_forecast(data, column, periods)
    
    def simple_forecast(self, data, column='Total_Revenue', periods=6):
        """Simple moving average forecast as fallback"""
        try:
            values = data[column].values
            forecast = []
            
            # Use weighted moving average
            for i in range(periods):
                if len(values) >= 4:
                    weights = [0.1, 0.2, 0.3, 0.4]  # More weight to recent values
                    window = values[-4:]
                    next_val = np.average(window, weights=weights[-len(window):]) * (1 + np.random.uniform(-0.05, 0.05))
                else:
                    next_val = values[-1] * (1 + np.random.uniform(-0.05, 0.05))
                forecast.append(next_val)
            
            return forecast
        except:
            return [data[column].iloc[-1]] * periods if len(data) > 0 else [0] * periods
    
    def calculate_forecast_metrics(self, actual, forecast):
        """Calculate forecast accuracy metrics"""
        if len(actual) != len(forecast):
            return {}
        
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

class LLMAssistant:
    def __init__(self):
        self.client = None
        self.setup_llm()
    
    def setup_llm(self):
        """Initialize Groq client"""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.client = groq.Groq(api_key=api_key)
            else:
                st.warning("GROQ_API_KEY not found in environment variables")
        except Exception as e:
            st.error(f"Error setting up LLM: {e}")
    
    def get_response(self, prompt, context=""):
        """Get response from Groq API"""
        if not self.client:
            return "LLM service not available. Please check your API key."
        
        try:
            full_prompt = f"""
            You are a supply chain analytics expert. Based on the following context about supply chain data, answer the user's question comprehensively and professionally.

            Context about the supply chain data:
            {context}

            User Question: {prompt}

            Please provide a detailed, analytical response focusing on supply chain insights, patterns, and recommendations.
            """
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.3,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting response: {e}"

def main():
    # Sidebar navigation with enhanced styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; color: white;">
        <h2 style="margin: 0; text-align: center;">🚀 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("", ["📊 Supply Chain Dashboard", "🤖 AI Research Assistant"])
    
    # Initialize classes
    data_path = "/home/y21tbh/Documents/supply-chain-ai/tsa-llm-supply-chain-project/data/data.csv"
    analyzer = SupplyChainAnalyzer(data_path)
    llm_assistant = LLMAssistant()
    
    if page == "📊 Supply Chain Dashboard":
        show_dashboard(analyzer)
    else:
        show_research_assistant(analyzer, llm_assistant)

def show_dashboard(analyzer):
    """Display the comprehensive supply chain dashboard"""
    
    st.markdown('<div class="main-header">🚀 Advanced Supply Chain Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if analyzer.df is None or analyzer.df.empty:
        st.error("No data loaded. Please check the data file path.")
        return
    
    # Enhanced metrics with better styling
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = analyzer.df['Total_Revenue'].sum() if 'Total_Revenue' in analyzer.df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Revenue</h3>
            <h2>${total_revenue:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_orders = len(analyzer.df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Orders</h3>
            <h2>{total_orders:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'Total_Lead_Time' in analyzer.df.columns:
            avg_lead_time = analyzer.df['Total_Lead_Time'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Lead Time</h3>
                <h2>{avg_lead_time:.1f} days</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Lead Time</h3>
                <h2>N/A</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'Sales_Channel' in analyzer.df.columns:
            unique_channels = analyzer.df['Sales_Channel'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Sales Channels</h3>
                <h2>{unique_channels}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>Sales Channels</h3>
                <h2>N/A</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Generate time series analysis
    analysis = analyzer.create_time_series_analysis()
    
    # Enhanced Time Series Analysis Section
    st.markdown('<div class="section-header">📊 Advanced Time Series Analysis</div>', unsafe_allow_html=True)
    
    if 'monthly_trends' in analysis and not analysis['monthly_trends'].empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue trend with enhanced styling
            fig1 = px.line(analysis['monthly_trends'], x='Order_Month', y='Total_Revenue',
                          title='📈 Monthly Revenue Trend Analysis',
                          template='plotly_white',
                          line_shape='spline')
            fig1.update_traces(line=dict(width=3))
            fig1.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Multi-metric comparison
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=analysis['monthly_trends']['Order_Month'], 
                                    y=analysis['monthly_trends']['Total_Revenue'],
                                    name='Revenue', line=dict(width=3, color='#1f77b4')))
            fig2.add_trace(go.Scatter(x=analysis['monthly_trends']['Order_Month'], 
                                    y=analysis['monthly_trends']['Quantity'],
                                    name='Quantity', line=dict(width=3, color='#ff7f0e'), yaxis='y2'))
            
            fig2.update_layout(
                title='📊 Revenue vs Quantity Trends',
                template='plotly_white',
                height=400,
                yaxis=dict(title='Revenue', side='left'),
                yaxis2=dict(title='Quantity', side='right', overlaying='y'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Advanced Forecasting Section
    st.markdown('<div class="section-header">🔮 Advanced Time Series Forecasting</div>', unsafe_allow_html=True)
    
    if 'monthly_trends' in analysis and not analysis['monthly_trends'].empty:
        # Generate multiple forecasts
        periods = 6
        
        arima_forecast = analyzer.arima_forecast(analysis['monthly_trends'], 'Total_Revenue', periods)
        holt_forecast = analyzer.holt_winters_forecast(analysis['monthly_trends'], 'Total_Revenue', periods)
        prophet_forecast = analyzer.prophet_style_forecast(analysis['monthly_trends'], 'Total_Revenue', periods)
        
        # Create forecast comparison visualization
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=analysis['monthly_trends']['Order_Month'],
            y=analysis['monthly_trends']['Total_Revenue'],
            name='Historical Revenue',
            line=dict(width=3, color='#1f77b4')
        ))
        
        # Forecasts
        forecast_months = [f'F{i+1}' for i in range(periods)]
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_months,
            y=arima_forecast,
            name='ARIMA Forecast',
            line=dict(width=2, color='#ff6b6b', dash='dash')
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_months,
            y=holt_forecast,
            name='Holt-Winters Forecast',
            line=dict(width=2, color='#2ecc71', dash='dash')
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_months,
            y=prophet_forecast,
            name='Prophet-Style Forecast',
            line=dict(width=2, color='#f39c12', dash='dash')
        ))
        
        fig_forecast.update_layout(
            title='🔮 Multi-Model Revenue Forecasting Comparison',
            template='plotly_white',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast metrics and insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="forecast-card">
                <h4>🔄 ARIMA Model</h4>
                <p>Autoregressive Integrated Moving Average - Captures trends and seasonality using autoregressive and moving average components</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="forecast-card">
                <h4>📈 Holt-Winters</h4>
                <p>Triple Exponential Smoothing - Handles trend and seasonality with exponential smoothing weights</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="forecast-card">
                <h4>🔮 Prophet-Style</h4>
                <p>Custom decomposition approach - Separates trend, seasonality, and holiday effects</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional Advanced Time Series Visualizations
    st.markdown('<div class="section-header">📈 Advanced Analytical Visualizations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal decomposition if enough data
        if 'monthly_trends' in analysis and len(analysis['monthly_trends']) >= 12:
            try:
                # Create time series for decomposition
                ts_data = analysis['monthly_trends'].set_index('Order_Month')['Total_Revenue']
                
                # Simple seasonal plot
                fig_seasonal = px.box(analyzer.df, 
                                    x=analyzer.df['Order_Date'].dt.month, 
                                    y='Total_Revenue',
                                    title='📅 Monthly Revenue Distribution (Seasonality)',
                                    template='plotly_white')
                fig_seasonal.update_layout(height=400)
                st.plotly_chart(fig_seasonal, use_container_width=True)
            except:
                st.info("Seasonal analysis requires more data points")
    
    with col2:
        # Cumulative revenue growth
        if 'monthly_trends' in analysis:
            cumulative_revenue = analysis['monthly_trends']['Total_Revenue'].cumsum()
            fig_cumulative = px.area(analysis['monthly_trends'], 
                                   x='Order_Month', 
                                   y=cumulative_revenue,
                                   title='📊 Cumulative Revenue Growth',
                                   template='plotly_white')
            fig_cumulative.update_layout(height=400)
            st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Channel and Warehouse Analysis
    st.markdown('<div class="section-header">🏢 Channel & Warehouse Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'channel_trends' in analysis and not analysis['channel_trends'].empty:
            fig4 = px.line(analysis['channel_trends'], x='Order_Month', y='Total_Revenue',
                          color='Sales_Channel', 
                          title='🛍️ Revenue by Sales Channel Over Time',
                          template='plotly_white')
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        if 'warehouse_trends' in analysis and not analysis['warehouse_trends'].empty:
            fig_warehouse = px.line(analysis['warehouse_trends'], x='Order_Month', y='Total_Revenue',
                                   color='Warehouse', 
                                   title='🏭 Revenue by Warehouse Over Time',
                                   template='plotly_white')
            fig_warehouse.update_layout(height=400)
            st.plotly_chart(fig_warehouse, use_container_width=True)
    
    # Enhanced Pattern Recognition Insights
    st.markdown('<div class="section-header">🔍 Advanced Pattern Recognition</div>', unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("📊 Statistical Insights")
        if 'monthly_trends' in analysis and not analysis['monthly_trends'].empty:
            revenue_growth = ((analysis['monthly_trends']['Total_Revenue'].iloc[-1] - 
                            analysis['monthly_trends']['Total_Revenue'].iloc[0]) / 
                           analysis['monthly_trends']['Total_Revenue'].iloc[0] * 100)
            st.write(f"**📈 Overall Revenue Growth:** {revenue_growth:.1f}%")
            
            volatility = analysis['monthly_trends']['Total_Revenue'].std() / analysis['monthly_trends']['Total_Revenue'].mean() * 100
            st.write(f"**📊 Revenue Volatility:** {volatility:.1f}%")
            
            st.write("**🔄 Trend Analysis:** Using ARIMA(2,1,2) for optimal balance between bias and variance")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🎯 Performance Metrics")
        if 'Total_Lead_Time' in analyzer.df.columns:
            lead_time_std = analyzer.df['Total_Lead_Time'].std()
            st.write(f"**⏱️ Lead Time Variability:** {lead_time_std:.1f} days std dev")
            
        if 'Sales_Channel' in analyzer.df.columns:
            best_channel = analyzer.df.groupby('Sales_Channel')['Total_Revenue'].sum().idxmax()
            st.write(f"**🏆 Top Performing Channel:** {best_channel}")
            
        st.write("**🔮 Forecast Horizon:** 6 periods ahead using multiple algorithms")
        st.markdown('</div>', unsafe_allow_html=True)

def show_research_assistant(analyzer, llm_assistant):
    """Display the AI research assistant page"""
    
    st.markdown('<div class="main-header">🤖 Advanced Supply Chain Research Assistant</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h3>🎯 About the AI Assistant</h3>
    <p>This AI assistant leverages advanced language models and time series analytics to provide deep insights into your supply chain data. 
    It combines statistical analysis with domain expertise to answer complex research questions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create enhanced context from the data
    context = ""
    if analyzer.df is not None and not analyzer.df.empty:
        context += f"## Dataset Overview:\n"
        context += f"- Total records: {len(analyzer.df)}\n"
        
        if 'Order_Date' in analyzer.df.columns:
            context += f"- Date range: {analyzer.df['Order_Date'].min()} to {analyzer.df['Order_Date'].max()}\n"
        
        if 'Sales_Channel' in analyzer.df.columns:
            channels = analyzer.df['Sales_Channel'].value_counts().to_dict()
            context += f"- Sales channels distribution: {channels}\n"
        
        if 'Warehouse' in analyzer.df.columns:
            warehouses = analyzer.df['Warehouse'].value_counts().to_dict()
            context += f"- Warehouses: {warehouses}\n"
        
        if 'Total_Revenue' in analyzer.df.columns:
            total_rev = analyzer.df['Total_Revenue'].sum()
            avg_rev = analyzer.df['Total_Revenue'].mean()
            context += f"- Total revenue: ${total_rev:,.2f}\n"
            context += f"- Average order value: ${avg_rev:,.2f}\n"
        
        # Add time series context
        analysis = analyzer.create_time_series_analysis()
        if 'monthly_trends' in analysis:
            context += f"- Monthly data points: {len(analysis['monthly_trends'])}\n"
            if len(analysis['monthly_trends']) > 1:
                growth_rate = ((analysis['monthly_trends']['Total_Revenue'].iloc[-1] - 
                              analysis['monthly_trends']['Total_Revenue'].iloc[0]) / 
                             analysis['monthly_trends']['Total_Revenue'].iloc[0] * 100)
                context += f"- Overall growth rate: {growth_rate:.1f}%\n"
    
    # Enhanced sample questions
    st.markdown("### 💡 Advanced Research Questions")
    sample_questions = [
        "Perform advanced time series decomposition and identify key seasonal patterns",
        "Analyze the impact of different forecasting models (ARIMA vs Holt-Winters) on inventory optimization",
        "What are the key drivers of supply chain variability and how can we mitigate them?",
        "Recommend advanced inventory optimization strategies using time series forecasting",
        "Analyze the correlation between order patterns and warehouse performance metrics"
    ]
    
    for i, question in enumerate(sample_questions):
        if st.button(f"🎯 Q{i+1}: {question}", key=f"q_{i}"):
            st.session_state.research_question = question
    
    # Enhanced research question input
    st.markdown("### 🔍 Ask an Advanced Research Question")
    research_question = st.text_area(
        "Enter your detailed research question:",
        value=st.session_state.get('research_question', ''),
        height=120,
        placeholder="e.g., Using ARIMA and exponential smoothing models, analyze the optimal inventory strategy for seasonal demand patterns and provide implementation recommendations..."
    )
    
    if st.button("🚀 Get Advanced AI Analysis", type="primary"):
        if research_question:
            with st.spinner("🔬 Performing advanced analysis with AI and statistical models..."):
                response = llm_assistant.get_response(research_question, context)
                
                st.markdown("### 📋 Advanced AI Analysis Results")
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.write(response)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a research question.")
    
    # Enhanced data context display
    with st.expander("📊 View Advanced Data Context Provided to AI"):
        st.markdown("### Data Context for AI Analysis")
        st.text(context)
        
        if analyzer.df is not None:
            st.markdown("### Sample Data")
            st.dataframe(analyzer.df.head(10))

if __name__ == "__main__":
    # Initialize session state
    if 'research_question' not in st.session_state:
        st.session_state.research_question = ""
    
    main()