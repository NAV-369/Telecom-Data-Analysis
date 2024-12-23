import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class ExperienceAnalytics:
    """Comprehensive user experience analysis for telecommunication data."""
    
    def __init__(self, xdr_data):
        """Initialize with telecom dataset."""
        self.xdr_data = xdr_data
    
    def _handle_outliers_and_missing(self, series):
        """
        Handle outliers and missing values by replacing with mean.
        
        Args:
            series (pd.Series): Input series
        
        Returns:
            pd.Series: Cleaned series
        """
        # Replace missing values with mean
        series_cleaned = series.fillna(series.mean())
        
        # Identify outliers using IQR method
        Q1 = series_cleaned.quantile(0.25)
        Q3 = series_cleaned.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with mean
        series_cleaned[(series_cleaned < lower_bound) | (series_cleaned > upper_bound)] = series_cleaned.mean()
        
        return series_cleaned
    
    def task_3_1_aggregate_customer_metrics(self):
        """
        Task 3.1: Aggregate metrics per customer.
        
        Returns:
            pd.DataFrame: Aggregated customer-level metrics
        """
        # Group by customer and aggregate
        customer_metrics = self.xdr_data.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Handset Type': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown',
            'Avg Bearer TP DL (kbps)': 'mean'
        }).reset_index()
        
        # Rename and clean columns
        customer_metrics.columns = [
            'Customer_ID', 
            'Avg_TCP_Retransmission', 
            'Avg_RTT', 
            'Handset_Type', 
            'Avg_Throughput'
        ]
        
        # Handle outliers and missing values
        for col in ['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']:
            customer_metrics[col] = self._handle_outliers_and_missing(customer_metrics[col])
        
        return customer_metrics
    
    def task_3_2_network_metrics_analysis(self):
        """
        Task 3.2: Compute top, bottom, and most frequent network metric values.
        
        Returns:
            dict: Detailed network metrics analysis
        """
        def get_metric_details(column):
            """Helper function to get metric details."""
            return {
                'Top_10': column.nlargest(10).tolist(),
                'Bottom_10': column.nsmallest(10).tolist(),
                'Most_Frequent_10': column.value_counts().head(10).index.tolist()
            }
        
        return {
            'TCP_Metrics': get_metric_details(self.xdr_data['TCP DL Retrans. Vol (Bytes)']),
            'RTT_Metrics': get_metric_details(self.xdr_data['Avg RTT DL (ms)']),
            'Throughput_Metrics': get_metric_details(self.xdr_data['Avg Bearer TP DL (kbps)'])
        }
    
    def task_3_3_handset_performance_analysis(self):
        """
        Task 3.3: Analyze performance metrics by handset type.
        
        Returns:
            dict: Performance metrics by handset type with interpretations
        """
        # Throughput analysis by handset type
        throughput_by_handset = self.xdr_data.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        
        # TCP retransmission analysis by handset type
        tcp_by_handset = self.xdr_data.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        
        # Prepare interpretations
        throughput_interpretation = (
            "Throughput Analysis:\n"
            "- Variations in mean throughput suggest different device capabilities.\n"
            "- High standard deviation indicates significant performance differences within handset types.\n"
            "- Compare mean and median to understand distribution skewness."
        )
        
        tcp_interpretation = (
            "TCP Retransmission Analysis:\n"
            "- Higher retransmission values indicate potential network connectivity issues.\n"
            "- Compare retransmission across handset types to identify potential device-specific problems.\n"
            "- Large standard deviations suggest inconsistent network performance."
        )
        
        return {
            'Throughput_By_Handset': throughput_by_handset,
            'TCP_By_Handset': tcp_by_handset,
            'Throughput_Interpretation': throughput_interpretation,
            'TCP_Interpretation': tcp_interpretation
        }
    
    def task_3_4_user_experience_clustering(self, n_clusters=3):
        """
        Task 3.4: Perform K-means clustering on user experience metrics.
        
        Args:
            n_clusters (int): Number of clusters to create
        
        Returns:
            dict: Clustering results with cluster descriptions
        """
        # Prepare data for clustering
        customer_metrics = self.task_3_1_aggregate_customer_metrics()
        
        # Select features for clustering
        clustering_features = ['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']
        X = customer_metrics[clustering_features]
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        customer_metrics['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze cluster characteristics
        cluster_summary = customer_metrics.groupby('Cluster')[clustering_features].mean()
        
        # Create detailed cluster descriptions
        cluster_descriptions = {}
        for i in range(n_clusters):
            cluster_data = cluster_summary.loc[i]
            description = (
                f"Cluster {i} Profile:\n"
                f"- TCP Retransmission: {cluster_data['Avg_TCP_Retransmission']:.2f} (Bytes)\n"
                f"- Average RTT: {cluster_data['Avg_RTT']:.2f} ms\n"
                f"- Average Throughput: {cluster_data['Avg_Throughput']:.2f} kbps\n"
                f"- Key Characteristics: {'High performance' if cluster_data['Avg_Throughput'] > 500 else 'Low performance'} "
                f"with {'stable' if cluster_data['Avg_TCP_Retransmission'] < 100 else 'unstable'} network"
            )
            cluster_descriptions[i] = description
        
        return {
            'Cluster_Summary': cluster_summary,
            'Cluster_Descriptions': cluster_descriptions,
            'Clustered_Data': customer_metrics
        }
