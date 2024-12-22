import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import DataPreparation
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Telecom Data Analysis",
    page_icon="📱",
    layout="wide"
)

# Initialize data preparation
@st.cache_data
def load_data():
    data_prep = DataPreparation()
    data_prep.load_data()
    return data_prep

# Main function
def main():
    st.title("📱 Telecom Data Analysis Dashboard")
    
    # Load data
    with st.spinner("Loading data..."):
        data_prep = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Overview", "Handset Analysis", "User Behavior", "Advanced Analytics"]
    )
    
    if page == "Overview":
        st.header("Overview")
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        
        total_users = data_prep.xdr_data['MSISDN/Number'].nunique()
        total_sessions = len(data_prep.xdr_data)
        
        # Convert data to numeric, replacing NULL values with 0
        dl_data = pd.to_numeric(
            data_prep.xdr_data['Total DL (Bytes)'].replace('\\N', '0'), 
            errors='coerce'
        ).fillna(0)
        
        ul_data = pd.to_numeric(
            data_prep.xdr_data['Total UL (Bytes)'].replace('\\N', '0'), 
            errors='coerce'
        ).fillna(0)
        
        total_data = (dl_data.sum() + ul_data.sum()) / (1024**3)  # Convert to GB
        
        col1.metric("Total Users", f"{total_users:,}")
        col2.metric("Total Sessions", f"{total_sessions:,}")
        col3.metric("Total Data (GB)", f"{total_data:.2f}")
        
        # Time series analysis
        st.subheader("Data Usage Over Time")
        
        # Handle NULL values and convert dates
        data_prep.xdr_data['Start_Clean'] = data_prep.xdr_data['Start'].replace('\\N', pd.NA)
        
        # Remove rows with invalid dates
        valid_dates_mask = data_prep.xdr_data['Start_Clean'].notna()
        clean_data = data_prep.xdr_data[valid_dates_mask].copy()
        
        # Convert to datetime with flexible parsing
        clean_data['Start_Date'] = pd.to_datetime(
            clean_data['Start_Clean'],
            format='mixed',
            errors='coerce'
        ).dt.date
        
        # Prepare daily usage data
        daily_usage = pd.DataFrame()
        daily_usage['Date'] = pd.date_range(
            start=clean_data['Start_Date'].min(),
            end=clean_data['Start_Date'].max()
        ).date
        
        # Calculate daily download and upload
        daily_dl = clean_data.groupby('Start_Date').agg({
            'Total DL (Bytes)': lambda x: pd.to_numeric(
                x.replace('\\N', '0'), 
                errors='coerce'
            ).sum()
        }).reset_index()
        
        daily_ul = clean_data.groupby('Start_Date').agg({
            'Total UL (Bytes)': lambda x: pd.to_numeric(
                x.replace('\\N', '0'), 
                errors='coerce'
            ).sum()
        }).reset_index()
        
        # Merge the data
        daily_usage = daily_usage.merge(
            daily_dl, 
            left_on='Date', 
            right_on='Start_Date', 
            how='left'
        ).merge(
            daily_ul,
            left_on='Date',
            right_on='Start_Date',
            how='left'
        )
        
        # Fill missing values with 0
        daily_usage['Total DL (Bytes)'] = daily_usage['Total DL (Bytes)'].fillna(0)
        daily_usage['Total UL (Bytes)'] = daily_usage['Total UL (Bytes)'].fillna(0)
        
        # Convert to GB
        daily_usage['Download (GB)'] = daily_usage['Total DL (Bytes)'] / (1024**3)
        daily_usage['Upload (GB)'] = daily_usage['Total UL (Bytes)'] / (1024**3)
        
        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_usage['Date'],
            y=daily_usage['Download (GB)'],
            name='Download (GB)',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=daily_usage['Date'],
            y=daily_usage['Upload (GB)'],
            name='Upload (GB)',
            line=dict(color='#ff7f0e')
        ))
        
        fig.update_layout(
            title="Daily Data Usage",
            xaxis_title="Date",
            yaxis_title="Data Volume (GB)",
            hovermode='x unified',
            showlegend=True
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Usage Statistics")
        col1, col2, col3 = st.columns(3)
        
        avg_daily_dl = daily_usage['Download (GB)'].mean()
        avg_daily_ul = daily_usage['Upload (GB)'].mean()
        peak_usage_day = daily_usage.loc[daily_usage['Download (GB)'].idxmax(), 'Date']
        
        col1.metric("Avg Daily Download", f"{avg_daily_dl:.2f} GB")
        col2.metric("Avg Daily Upload", f"{avg_daily_ul:.2f} GB")
        col3.metric("Peak Usage Date", peak_usage_day.strftime('%Y-%m-%d'))
        
    elif page == "Handset Analysis":
        st.header("Handset Analysis")
        
        # Get handset data
        top_10_handsets, top_3_manufacturers, top_5_per_manufacturer = data_prep.analyze_handsets()
        
        # Display top manufacturers
        st.subheader("Top Manufacturers")
        fig = px.bar(
            top_3_manufacturers,
            x='manufacturer',
            y='count',
            title="Top 3 Manufacturers"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top handsets
        st.subheader("Top Handsets")
        fig = px.bar(
            top_10_handsets,
            x='handset',
            y='count',
            title="Top 10 Handsets"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top handsets per manufacturer
        st.subheader("Top Handsets by Manufacturer")
        selected_manufacturer = st.selectbox(
            "Select Manufacturer",
            options=top_3_manufacturers['manufacturer'].tolist()
        )
        
        if selected_manufacturer in top_5_per_manufacturer:
            fig = px.bar(
                top_5_per_manufacturer[selected_manufacturer],
                x='handset',
                y='count',
                title=f"Top 5 Handsets - {selected_manufacturer}"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
    elif page == "User Behavior":
        st.header("User Behavior Analysis")
        
        # Get user metrics
        user_metrics = data_prep.aggregate_user_behavior()
        
        # Data usage distribution
        st.subheader("Data Usage Distribution")
        
        # Convert to GB for better visualization
        user_metrics['total_data_volume_gb'] = user_metrics['total_data_volume'] / (1024**3)
        
        fig = px.histogram(
            user_metrics,
            x="total_data_volume_gb",
            nbins=50,
            title="Distribution of Total Data Usage (GB)",
            labels={"total_data_volume_gb": "Total Data Volume (GB)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Application usage
        st.subheader("Application Usage")
        app_cols = [col for col in user_metrics.columns if col.startswith('total_data_volume_') 
                   and col != 'total_data_volume']
        
        # Convert all values to GB and calculate total
        app_usage_gb = user_metrics[app_cols].sum() / (1024**3)
        app_usage_gb.index = [col.replace('total_data_volume_', '').title() 
                            for col in app_usage_gb.index]
        
        # Create pie chart
        fig = px.pie(
            values=app_usage_gb.values,
            names=app_usage_gb.index,
            title="Data Usage by Application (GB)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # User segments
        st.subheader("User Segments")
        
        # Clean and prepare data for segmentation
        user_metrics_clean = data_prep.handle_outliers(
            user_metrics,
            ['total_data_volume', 'duration', 'session_id']
        )
        user_segments = data_prep.create_deciles(user_metrics_clean, 'total_data_volume')
        
        # Calculate segment statistics
        segment_stats = user_segments.groupby('decile_class').agg({
            'total_data_volume': ['mean', 'count'],
            'session_id': 'mean',
            'duration': 'mean'
        })
        
        # Convert data volume to GB and duration to hours
        segment_stats[('total_data_volume', 'mean')] = segment_stats[('total_data_volume', 'mean')] / (1024**3)
        segment_stats[('duration', 'mean')] = segment_stats[('duration', 'mean')] / (1000 * 60 * 60)  # ms to hours
        
        # Rename columns for better display
        segment_stats.columns = [
            'Avg Data Usage (GB)',
            'Number of Users',
            'Avg Sessions per User',
            'Avg Usage Duration (hours)'
        ]
        
        # Round values for better display
        segment_stats = segment_stats.round(2)
        
        # Display segment statistics
        st.dataframe(segment_stats, use_container_width=True)
        
    else:  # Advanced Analytics
        st.header("Advanced Analytics")
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        user_metrics = data_prep.aggregate_user_behavior()
        numeric_columns = ['session_id', 'duration', 'download_data', 'upload_data', 'total_data_volume']
        correlation_matrix = data_prep.compute_correlation_matrix(user_metrics, numeric_columns)
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # PCA analysis
        st.subheader("Principal Component Analysis")
        pca_columns = ['session_id', 'duration', 'download_data', 'upload_data']
        pca_results, explained_variance = data_prep.perform_pca(user_metrics, pca_columns)
        
        fig = px.scatter(
            pca_results,
            x='PC1',
            y='PC2',
            title=f"PCA Results (Explained Variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%})"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
