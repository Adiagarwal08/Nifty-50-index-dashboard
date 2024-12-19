import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from io import BytesIO
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist

st.title('Nifty 50 Analysis')


# Get Nifty 50 tickers
nifty50_tickers = ['HDFCBANK.NS', 'ICICIBANK.NS', 'TCS.NS', 'RELIANCE.NS', 'INFY.NS',
                   'HCLTECH.NS', 'KOTAKBANK.NS', 'LT.NS', 'SBIN.NS', 'ITC.NS',
                   'AXISBANK.NS', 'HDFC.NS', 'BAJFINANCE.NS', 'MARUTI.NS', 'NTPC.NS',
                   'BPCL.NS', 'BHARTIARTL.NS', 'POWERGRID.NS', 'BAJAJFINSV.NS', 'HINDUNILVR.NS',
                   'ADANIPORTS.NS', 'WIPRO.NS', 'TATAMOTORS.NS', 'COALINDIA.NS', 'DIVISLAB.NS',
                   'SUNPHARMA.NS', 'DRREDDY.NS', 'TECHM.NS', 'ONGC.NS', 'CIPLA.NS',
                   'HEROMOTOCO.NS', 'SBILIFE.NS', 'BRITANNIA.NS', 'JSWSTEEL.NS', 'TATAMOTORS.NS',
                   'NESTLEIND.NS', 'ULTRACEMCO.NS', 'HINDALCO.NS', 'TITAN.NS', 'M&M.NS',
                   'UPL.NS', 'ASIANPAINT.NS', 'HDFCLIFE.NS', 'SBIN.NS']

# Sidebar for selections
with st.sidebar:
    # Select a ticker from the dropdown (optional)
    selected_ticker = st.selectbox('Select a Nifty 50 Stock', nifty50_tickers + [""], index=len(nifty50_tickers))

    # Get start and end dates
    # start_date = st.date_input("Start Start Date")
    # end_date = st.date_input("Select End date", datetime.today())  # Default to today
    # Default to 2024 data
    start_date = date(2024, 1, 1)
    end_date = date.today()

    # User input for dates
    start_date = st.date_input("Start Date", start_date)
    end_date = st.date_input("End Date", end_date)

# Calculate date range difference in days
date_diff_days = (end_date - start_date).days

# Determine interval based on date range difference
if date_diff_days <= 100:
    interval = "1d"  # Daily
elif date_diff_days <= 380:
    interval = "1wk"  # Weekly
else:
    interval = "1mo"  # Monthly

# Download data for Nifty 50 index for the year 2023 (fixed year for comparison)
nifty50_start_date = datetime(2023, 1, 1)
nifty50_end_date = datetime(2023, 12, 31)
nifty50_data_week = yf.download("^NSEI", start=nifty50_start_date, end=nifty50_end_date, interval="1wk")
nifty50_data_days = yf.download("^NSEI", start=nifty50_start_date, end=nifty50_end_date, interval="1d")

# Download data for selected ticker (if user chose one)
if selected_ticker:
    stock_data = yf.download(selected_ticker, start=start_date, end=end_date, interval=interval)
else:
    stock_data = None

# Create candlestick charts
fig_nifty50 = go.Figure(data=[go.Candlestick(x=nifty50_data_week.index,
                                             open=nifty50_data_week['Open'],
                                             high=nifty50_data_week['High'],
                                             low=nifty50_data_week['Low'],
                                             close=nifty50_data_week['Close'],
                                             name='Nifty 50')])
fig_nifty50.update_layout(
    title=f"Nifty 50 Candlestick Chart (2023)",
    xaxis_rangeslider_visible=False,
    xaxis_title="Date",
    yaxis_title="Price",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

# Calculate closing prices and log returns for Nifty 50 (2023)
nifty50_closing_prices = nifty50_data_days['Close']
nifty50_log_returns = np.log(nifty50_closing_prices / nifty50_closing_prices.shift(1))

# Create a figure for closing prices
fig_closing_prices = go.Figure()
fig_closing_prices.add_trace(go.Scatter(x=nifty50_data_days.index, y=nifty50_closing_prices, mode='lines', name='Nifty 50 Closing Prices'))
fig_closing_prices.update_layout(
    title='Nifty 50 Closing Prices (2023)',
    xaxis_title='Date',
    yaxis_title='Price',
    font=dict(
        family='Courier New, monospace',
        size=18,
        color='#7f7f7f'
    )
)

# Create a figure for log returns
fig_log_returns = go.Figure()
fig_log_returns.add_trace(go.Scatter(x=nifty50_data_days.index, y=nifty50_log_returns, mode='lines', name='Nifty 50 Log Returns'))
fig_log_returns.update_layout(
    title='Nifty 50 Log Returns (2023)',
    xaxis_title='Date',
    yaxis_title='Log Return',
    font=dict(
        family='Courier New, monospace',
        size=18,
        color='#7f7f7f'
    )
)

# Download stock data based on user selection (assuming a ticker is chosen)
try:
    live_stock_data = yf.download("^NSEI", start=start_date, end=end_date, interval=interval)
except (yf.DownloadError, ValueError) as e:
    st.error(f"Error downloading data: {e}")
    live_stock_data = None  # Handle the error gracefully or display an empty chart

if live_stock_data is not None:
    # Create candlestick chart
    live_fig_stock = go.Figure(data=[go.Candlestick(
        x=live_stock_data.index,
        open=live_stock_data['Open'],
        high=live_stock_data['High'],
        low=live_stock_data['Low'],
        close=live_stock_data['Close'],

        name="Nifty50")])

    live_fig_stock.update_layout(
        title=f"Nifty50 Live Candlestick Chart",
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price",
        font=dict(family="Courier New, monospace", size=18, color="#7f7f7f")
    )

    st.plotly_chart(live_fig_stock)

    # Display the closing price on the end date
    end_date_price = live_stock_data['Close'].iloc[-1]
    st.write(f"Closing Price on {end_date}: {end_date_price:.2f}")



# Display the charts
st.plotly_chart(fig_nifty50)
st.plotly_chart(fig_closing_prices)
st.plotly_chart(fig_log_returns)

if stock_data is not None:
    fig_stock = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                 open=stock_data['Open'],
                                                 high=stock_data['High'],
                                                 low=stock_data['Low'],
                                                 close=stock_data['Close'],
                                                 name=selected_ticker)])
    fig_stock.update_layout(
        title=f"{selected_ticker} Candlestick Chart",
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title="Price",
        font=dict(family="Courier New, monospace",
                  size=18,
                  color="#7f7f7f"
                  )
    )

    st.plotly_chart(fig_stock)




# Fetch Nifty 50 data
nifty_ticker = "^NSEI"  # Nifty 50 ticker symbol
nifty_data = yf.download(nifty_ticker, start='2023-01-01', end='2023-12-31')

if not nifty_data.empty:
    nifty_data['Log Return'] = np.log(nifty_data['Close'] / nifty_data['Close'].shift(1))
    nifty_data.dropna(inplace=True)


    # Calculate correlation and distance matrices
    nifty_corr_matrix = nifty_data[['Close', 'Log Return']].corr()
    nifty_dist_matrix = pairwise_distances(nifty_data[['Close', 'Log Return']])
        

    # Multidimensional Scaling (MDS) plot
    mds_nifty = MDS(n_components=2, dissimilarity="precomputed")
    mds_nifty_transformed = mds_nifty.fit_transform(nifty_dist_matrix)

    # Optimal KMeans Clustering
    st.subheader("Nifty 50 KMeans Clustering with Elbow Method")
    inertia_nifty = []
    K_nifty = range(1, 11)
    for k in K_nifty:
        kmeans_nifty = KMeans(n_clusters=k, random_state=0).fit(mds_nifty_transformed)
        inertia_nifty.append(kmeans_nifty.inertia_)
    fig_elbow_nifty = plt.figure(figsize=(6, 4))
    plt.plot(K_nifty, inertia_nifty, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K (Nifty 50)')
    st.pyplot(fig_elbow_nifty)

    # Clustering with optimal number of clusters (manually chosen here as an example)
    optimal_clusters_nifty = 3
    kmeans_optimal_nifty = KMeans(n_clusters=optimal_clusters_nifty, random_state=0).fit(mds_nifty_transformed)
    fig_clusters_nifty = px.scatter(x=mds_nifty_transformed[:, 0], y=mds_nifty_transformed[:, 1], color=kmeans_optimal_nifty.labels_.astype(str),
                                     title="Nifty 50 KMeans Clustering", labels={'x': 'MDS1', 'y': 'MDS2', 'color': 'Cluster'}
)



# Function to get Nifty 50 symbols with sectors
@st.cache_data
def get_nifty50_symbols():
    return {
        'ADANIPORTS.NS': 'Infrastructure', 'ASIANPAINT.NS': 'Consumer Goods', 
        'AXISBANK.NS': 'Financial Services', 'BAJAJ-AUTO.NS': 'Automobile',
        'BAJFINANCE.NS': 'Financial Services', 'BAJAJFINSV.NS': 'Financial Services',
        'BPCL.NS': 'Energy', 'BHARTIARTL.NS': 'Telecom', 'BRITANNIA.NS': 'Consumer Goods',
        'CIPLA.NS': 'Healthcare', 'COALINDIA.NS': 'Energy', 'DIVISLAB.NS': 'Healthcare',
        'DRREDDY.NS': 'Healthcare', 'EICHERMOT.NS': 'Automobile', 'GRASIM.NS': 'Manufacturing',
        'HCLTECH.NS': 'IT', 'HDFCBANK.NS': 'Financial Services', 'HDFCLIFE.NS': 'Financial Services',
        'HEROMOTOCO.NS': 'Automobile', 'HINDALCO.NS': 'Metal', 'HINDUNILVR.NS': 'Consumer Goods',
        'ICICIBANK.NS': 'Financial Services', 'ITC.NS': 'Consumer Goods',
        'INDUSINDBK.NS': 'Financial Services', 'INFY.NS': 'IT', 'JSWSTEEL.NS': 'Metal',
        'KOTAKBANK.NS': 'Financial Services', 'LT.NS': 'Infrastructure',
        'M&M.NS': 'Automobile', 'MARUTI.NS': 'Automobile', 'NTPC.NS': 'Energy',
        'NESTLEIND.NS': 'Consumer Goods', 'ONGC.NS': 'Energy', 'POWERGRID.NS': 'Energy',
        'RELIANCE.NS': 'Energy', 'SBILIFE.NS': 'Financial Services',
        'SBIN.NS': 'Financial Services', 'SUNPHARMA.NS': 'Healthcare',
        'TCS.NS': 'IT', 'TATACONSUM.NS': 'Consumer Goods', 'TATAMOTORS.NS': 'Automobile',
        'TATASTEEL.NS': 'Metal', 'TECHM.NS': 'IT', 'TITAN.NS': 'Consumer Goods',
        'UPL.NS': 'Chemical', 'ULTRACEMCO.NS': 'Manufacturing', 'WIPRO.NS': 'IT',
        'APOLLOHOSP.NS': 'Healthcare', 'ADANIENT.NS': 'Infrastructure', 'DMART.NS': 'Retail'
    }

# Function to fetch data for a single symbol
def fetch_single_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if not stock_data.empty:
            return stock_data['Close']
        return None
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return None

# Function to fetch and process all stock data
@st.cache_data
def fetch_stock_data():
    symbols = get_nifty50_symbols()
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    data_dict = {}
    progress_bar = st.progress(0)
    
    for idx, (symbol, sector) in enumerate(symbols.items()):
        closing_prices = fetch_single_stock_data(symbol, start_date, end_date)
        if closing_prices is not None:
            data_dict[symbol.replace('.NS', '')] = closing_prices
        progress = (idx + 1) / len(symbols)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return pd.DataFrame(data_dict)

# Function to calculate returns
def calculate_returns(df):
    return df.pct_change().dropna()

# Function to create correlation heatmap
def create_correlation_heatmap(data):
    correlation_matrix = data.corr()
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    sns.heatmap(correlation_matrix, 
                annot=True,
                cmap='RdYlBu',
                vmin=-1, 
                vmax=1,
                center=0,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Nifty 50 Stocks Correlation Heatmap (2023)', pad=20)
    plt.tight_layout()
    
    return fig, correlation_matrix

# Function to create distance matrix heatmap
def create_distance_matrix_heatmap(data):
    returns = calculate_returns(data)
    distances = pdist(returns.T, metric='euclidean')
    distance_matrix = squareform(distances)
    
    distance_df = pd.DataFrame(
        distance_matrix,
        index=data.columns,
        columns=data.columns
    )
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    sns.heatmap(distance_df,
                annot=True,
                cmap='viridis_r',
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Euclidean Distance'},
                ax=ax)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Nifty 50 Stocks Distance Matrix Heatmap (2023)', pad=20)
    plt.tight_layout()
    
    return fig, distance_df


# Function to create 3D MDS plot
def create_3d_mds_plot(data):
    returns = calculate_returns(data)
    corr_matrix = returns.corr()
    distance_matrix = 1 - corr_matrix
    
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(distance_matrix)
    
    symbols_dict = get_nifty50_symbols()
    sectors = [symbols_dict[f"{stock}.NS"] for stock in data.columns]
    
    mds_df = pd.DataFrame({
        'Stock': data.columns,
        'X': mds_coords[:, 0],
        'Y': mds_coords[:, 1],
        'Z': mds_coords[:, 2],
        'Sector': sectors
    })
    
    fig = px.scatter_3d(
        mds_df,
        x='X',
        y='Y',
        z='Z',
        color='Sector',
        text='Stock',
        labels={'X': 'First Dimension', 'Y': 'Second Dimension', 'Z': 'Third Dimension'}
    )
    
    fig.update_traces(
        marker=dict(size=8),
        textposition='top center',
        hovertemplate="<br>".join([
            "Stock: %{text}",
            "Sector: %{marker.color}",
            "<extra></extra>"
        ])
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='MDS Dimension 1',
            yaxis_title='MDS Dimension 2',
            zaxis_title='MDS Dimension 3'
        ),
        height=800
    )
    
    return fig


def create_3d_kmeans_plot(data, n_clusters=5):
    # Calculate returns
    returns = calculate_returns(data)
    
    # Standardize the returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns.T)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_returns)
    
    # Apply MDS for 3D visualization
    mds = MDS(n_components=3, random_state=42)
    mds_coords = mds.fit_transform(scaled_returns)
    
    # Create DataFrame for plotting
    symbols_dict = get_nifty50_symbols()
    sectors = [symbols_dict[f"{stock}.NS"] for stock in data.columns]
    
    cluster_df = pd.DataFrame({
        'Stock': data.columns,
        'X': mds_coords[:, 0],
        'Y': mds_coords[:, 1],
        'Z': mds_coords[:, 2],
        'Sector': sectors,
        'Cluster': [f'Cluster {i}' for i in clusters]
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        cluster_df,
        x='X',
        y='Y',
        z='Z',
        color='Cluster',
        symbol='Sector',
        text='Stock',
        title='3D K-means Clustering of Nifty 50 Stocks',
        labels={'X': 'MDS Dimension 1', 'Y': 'MDS Dimension 2', 'Z': 'MDS Dimension 3'}
    )
    
    fig.update_traces(
        marker=dict(size=10),
        textposition='top center',
        hovertemplate="<br>".join([
            "Stock: %{text}",
            "Cluster: %{color}",
            "Sector: %{marker.symbol}",
            "<extra></extra>"
        ])
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='MDS Dimension 1',
            yaxis_title='MDS Dimension 2',
            zaxis_title='MDS Dimension 3'
        ),
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Add this function to calculate optimal number of clusters
def find_optimal_clusters(data, max_clusters=10):
    returns = calculate_returns(data)
    scaled_returns = StandardScaler().fit_transform(returns.T)
    
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_returns)
        score = silhouette_score(scaled_returns, clusters)
        silhouette_scores.append(score)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Create plot of silhouette scores
    fig = px.line(
        x=list(range(2, max_clusters + 1)),
        y=silhouette_scores,
        title='Silhouette Score vs Number of Clusters',
        labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'}
    )
    
    return optimal_clusters, fig

def load_data():
    sector_closing_data = pd.read_csv(r'C:\Users\Aditya Agarwal\OneDrive\Desktop\Semester\Semester-5\Data Science in Financial Markets\dashboard\close_price.csv', parse_dates=['Date'])
    sector_closing_data.set_index('Date', inplace=True)
    return sector_closing_data

def load_log_returns():
    sector_log_returns = pd.read_csv(r'C:\Users\Aditya Agarwal\OneDrive\Desktop\Semester\Semester-5\Data Science in Financial Markets\dashboard\log_returns.csv', parse_dates=['Date'])
    sector_log_returns.set_index('Date', inplace=True)
    return sector_log_returns

# Helper function for downloading figures as PNG
def download_figure(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer

def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

# Main app
def main():
    nifty50_index = yf.Ticker("^NSEI")
    nifty50_data = nifty50_index.history(start=start_date, end=end_date)

    # Load the data
    data = load_data()
    log_returns = load_log_returns()

    # Get the list of stock names from the column headers
    stock_options = data.columns.tolist()

    # Create a sidebar with a multiple select option
    selected_stocks = st.sidebar.multiselect("Select Stocks", stock_options)

    # Filter the data for the selected stocks
    filtered_data = data[selected_stocks]
    filtered_log_returns = log_returns[selected_stocks]

    

    # Create line plot for closing prices
    fig_price = px.line(filtered_data, x=filtered_data.index, y=selected_stocks,
                        labels={'index': 'Date', 'value': 'Closing Price'},
                        title='Closing Price of Selected Stocks')
    fig_price.update_xaxes(tickformat="%m-%Y")
    fig_price.update_layout(xaxis_rangeslider_visible=False)

    # Create line plot for log returns
    fig_log_returns = px.line(filtered_log_returns, x=filtered_log_returns.index, y=selected_stocks,
                              labels={'index': 'Date', 'value': 'Log Returns'},
                              title='Log Returns of Selected Stocks')
    fig_log_returns.update_xaxes(tickformat="%m-%Y")
    fig_log_returns.update_layout(xaxis_rangeslider_visible=False)

    # Display the plots
    st.plotly_chart(fig_price)
    st.plotly_chart(fig_log_returns)

    # Distribution analysis (Fat tails)
    st.subheader("Distribution Analysis for Fat Tails")
    selected_stock_dist = st.selectbox("Select Stock for Distribution Analysis:", selected_stocks)

    # Check for valid selection
    if selected_stock_dist:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_log_returns[selected_stock_dist].dropna(), kde=True, ax=ax)
        ax.set(title=f"Distribution of Log Returns for {selected_stock_dist}", xlabel="Log Returns", ylabel="Frequency")
        st.pyplot(fig)
        
        # Download button for the distribution chart
        st.download_button(
            "Download Distribution Chart as PNG", 
            data=download_figure(fig), 
            file_name=f"distribution_{selected_stock_dist}.png", 
            mime="image/png"
        )
    else:
        st.warning("Please select a valid stock for distribution analysis.")

    # Autocorrelation analysis
    st.subheader("Autocorrelation Analysis")
    selected_stock_auto = st.selectbox("Select Stock for Autocorrelation Analysis:", selected_stocks)

    if selected_stock_auto:
        fig, ax = plt.subplots(figsize=(10, 5))
        pd.plotting.autocorrelation_plot(filtered_log_returns[selected_stock_auto].dropna(), ax=ax)
        ax.set(title=f"Autocorrelation of {selected_stock_auto} Log Returns")
        st.pyplot(fig)
        
        # Download button for the autocorrelation chart
        st.download_button(
            "Download Autocorrelation Chart as PNG", 
            data=download_figure(fig), 
            file_name=f"autocorrelation_{selected_stock_auto}.png", 
            mime="image/png"
        )
    else:
        st.warning("Please select a valid stock for autocorrelation analysis.")

    # Heatmap type selection
    heatmap_type = st.radio("Select Heatmap Type:", ["Correlation Matrix", "Distance Matrix"], index=0, horizontal=True)
    
    if heatmap_type == "Correlation Matrix":
        st.subheader("Correlation Matrix Heatmap")
        corr_matrix = filtered_log_returns.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set(title="Correlation Matrix Heatmap")
        st.pyplot(fig)
        st.download_button("Download Correlation Matrix as PNG", data=download_figure(fig), file_name="correlation_matrix.png", mime="image/png")
    else:
        st.subheader("Distance Matrix Heatmap")
        dist_matrix = pairwise_distances(filtered_log_returns.T, metric="euclidean")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(dist_matrix, cmap="viridis", xticklabels=selected_stocks, yticklabels=selected_stocks, ax=ax)
        ax.set(title="Distance Matrix Heatmap")
        st.pyplot(fig)
        st.download_button("Download Distance Matrix as PNG", data=download_figure(fig), file_name="distance_matrix.png", mime="image/png")

    # Multidimensional Scaling (MDS)
    dist_matrix = pairwise_distances(filtered_log_returns.T, metric="euclidean")
    st.subheader("Multidimensional Scaling (MDS)")
    if len(selected_stocks) > 1:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(dist_matrix)
        mds_df = pd.DataFrame(mds_coords, index=selected_stocks, columns=['MDS1', 'MDS2'])
        fig_mds = px.scatter(mds_df, x='MDS1', y='MDS2', text=mds_df.index, title="Multidimensional Scaling (MDS)")
        fig_mds.update_traces(textposition="top center")
        st.plotly_chart(fig_mds)
    else:
        st.warning("At least 2 stocks must be selected for MDS visualization.")

    # Elbow method for optimal clusters
    st.subheader("Optimal Number of Clusters (Elbow Method)")
    inertia_values = []
    k_range = range(1, min(11, len(selected_stocks) + 1))

    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        if len(selected_stocks) >= k:
            kmeans_temp.fit(dist_matrix)
            inertia_values.append(kmeans_temp.inertia_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_range, inertia_values, marker='o', linestyle='--')
    ax.set(title="Elbow Method for Optimal k", xlabel="Number of Clusters (k)", ylabel="Inertia")
    st.pyplot(fig)
    st.download_button("Download Elbow Method Plot as PNG", data=download_figure(fig), file_name="elbow_method.png", mime="image/png")

    # KMeans clustering
    st.subheader("KMeans Clustering")
    max_k = min(10, len(selected_stocks) - 1)

    if len(selected_stocks) > 1:
        if max_k > 1:
            optimal_k = st.slider("Select number of clusters (k):", 2, max_k, min(4, max_k))
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(dist_matrix)
            mds_df['Cluster'] = clusters
            fig_kmeans = px.scatter(mds_df, x='MDS1', y='MDS2', color='Cluster', text=mds_df.index, title=f"KMeans Clustering with k={optimal_k}")
            fig_kmeans.update_traces(textposition="top center")
            st.plotly_chart(fig_kmeans)
        else:
            st.warning("Not enough stocks to form multiple clusters. Please select more stocks.")
    else:
        st.warning("At least 2 stocks must be selected to perform clustering.")

    # Download options for data
    st.subheader("Download Data")
    csv_data = convert_df_to_csv(filtered_data)
    st.download_button(label="Download Stock Data as CSV", data=csv_data, file_name='stock_data.csv', mime='text/csv')


    tab1, tab2, tab3 = st.tabs(["3D MDS Plot", "Correlation Heatmap", "Distance Matrix"])
    
    with st.spinner('Fetching stock data... This may take a few minutes...'):
        df = fetch_stock_data()
    
    if df.empty:
        st.error("No data was fetched. Please check your internet connection and try again.")
        return
    
    
    with tab1:
        st.header("3D MDS Analysis")
        
        try:
            fig_mds = create_3d_mds_plot(df)
            st.plotly_chart(fig_mds, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating 3D MDS plot: {e}")
    
    with tab2:
        st.header("Correlation Analysis")
       
        
        try:
            fig_corr, corr_matrix = create_correlation_heatmap(df)
            st.pyplot(fig_corr)
            
            # Add download button for correlation matrix
            csv_corr = corr_matrix.to_csv()
            st.download_button(
                label="Download Correlation Matrix as CSV",
                data=csv_corr,
                file_name="nifty50_correlation_matrix.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")
    
    with tab3:
        st.header("Distance Matrix Analysis")
        
        
        try:
            fig_dist, dist_df = create_distance_matrix_heatmap(df)
            st.pyplot(fig_dist)
            
            # Add download button for distance matrix
            csv_dist = dist_df.to_csv()
            st.download_button(
                label="Download Distance Matrix as CSV",
                data=csv_dist,
                file_name="nifty50_distance_matrix.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error creating distance matrix heatmap: {e}")

if __name__ == "__main__":
    main()
