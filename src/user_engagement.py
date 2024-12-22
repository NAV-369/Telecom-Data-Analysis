import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class UserEngagementAnalyzer:
    """Class for analyzing user engagement metrics and clustering users."""
    
    def __init__(self, xdr_data: pd.DataFrame):
        """Initialize with XDR data."""
        self.xdr_data = xdr_data
        self.engagement_metrics = None
        self.clusters = None
        
    def aggregate_engagement_metrics(self) -> pd.DataFrame:
        """Aggregate engagement metrics per customer."""
        # First, let's convert the numeric columns, handling the '\\N' values
        numeric_cols = ['Dur. (ms)', 'DL (Bytes)', 'UL (Bytes)']
        df = self.xdr_data.copy()
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].replace('\\N', '0'), errors='coerce')
        
        # Aggregate metrics per customer
        metrics = df.groupby('MSISDN').agg({
            'Bearer Id': 'count',  # session frequency
            'Dur. (ms)': 'sum',    # total duration
            'DL (Bytes)': 'sum',   # total download
            'UL (Bytes)': 'sum'    # total upload
        }).reset_index()
        
        # Calculate total traffic
        metrics['Total Traffic (Bytes)'] = metrics['DL (Bytes)'] + metrics['UL (Bytes)']
        
        # Rename columns for clarity
        metrics.columns = ['MSISDN', 'Session_Frequency', 'Duration_ms', 
                         'Download_Bytes', 'Upload_Bytes', 'Total_Traffic']
        
        self.engagement_metrics = metrics
        return metrics
    
    def get_top_users(self, metric: str, n: int = 10) -> pd.DataFrame:
        """Get top n users for a specific metric."""
        if self.engagement_metrics is None:
            self.aggregate_engagement_metrics()
            
        return self.engagement_metrics.nlargest(n, metric)
    
    def normalize_metrics(self) -> pd.DataFrame:
        """Normalize engagement metrics using StandardScaler."""
        if self.engagement_metrics is None:
            self.aggregate_engagement_metrics()
            
        metrics_to_normalize = ['Session_Frequency', 'Duration_ms', 'Total_Traffic']
        scaler = StandardScaler()
        
        normalized_data = self.engagement_metrics.copy()
        normalized_data[metrics_to_normalize] = scaler.fit_transform(
            normalized_data[metrics_to_normalize]
        )
        
        return normalized_data
    
    def perform_kmeans(self, k: int = 3) -> Tuple[pd.DataFrame, Dict]:
        """Perform k-means clustering on normalized metrics."""
        normalized_data = self.normalize_metrics()
        
        # Select features for clustering
        features = ['Session_Frequency', 'Duration_ms', 'Total_Traffic']
        X = normalized_data[features]
        
        # Perform k-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Add cluster labels to the original data
        self.engagement_metrics['Cluster'] = clusters
        
        # Compute cluster statistics
        cluster_stats = {}
        for i in range(k):
            cluster_data = self.engagement_metrics[self.engagement_metrics['Cluster'] == i]
            cluster_stats[f'Cluster_{i}'] = {
                'size': len(cluster_data),
                'min': cluster_data[features].min(),
                'max': cluster_data[features].max(),
                'mean': cluster_data[features].mean(),
                'total': cluster_data[features].sum()
            }
        
        return self.engagement_metrics, cluster_stats
    
    def find_optimal_k(self, max_k: int = 10) -> Tuple[List[int], List[float]]:
        """Find optimal k using elbow method."""
        normalized_data = self.normalize_metrics()
        features = ['Session_Frequency', 'Duration_ms', 'Total_Traffic']
        X = normalized_data[features]
        
        inertias = []
        silhouette_scores = []
        k_values = range(2, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        return list(k_values), inertias, silhouette_scores
    
    def analyze_app_engagement(self) -> pd.DataFrame:
        """Analyze user engagement per application."""
        # Get application columns
        app_columns = [col for col in self.xdr_data.columns 
                      if 'Social Media' in col or 'Google' in col or 
                      'Email' in col or 'Youtube' in col or 
                      'Netflix' in col or 'Gaming' in col]
        
        # Aggregate traffic per user and application
        app_engagement = pd.DataFrame()
        for app in app_columns:
            user_app_traffic = self.xdr_data.groupby('MSISDN').agg({
                app: lambda x: pd.to_numeric(x.replace('\\N', '0'), errors='coerce').sum()
            }).reset_index()
            
            # Get top 10 users for this app
            top_users = user_app_traffic.nlargest(10, app)
            app_engagement[f'{app}_Top_Users'] = top_users['MSISDN']
            app_engagement[f'{app}_Traffic'] = top_users[app]
        
        return app_engagement
    
    def plot_top_apps(self) -> None:
        """Plot top 3 most used applications."""
        # Calculate total traffic per application
        app_columns = [col for col in self.xdr_data.columns 
                      if 'Social Media' in col or 'Google' in col or 
                      'Email' in col or 'Youtube' in col or 
                      'Netflix' in col or 'Gaming' in col]
        
        app_traffic = {}
        for app in app_columns:
            total_traffic = pd.to_numeric(
                self.xdr_data[app].replace('\\N', '0'), 
                errors='coerce'
            ).sum()
            app_traffic[app] = total_traffic
        
        # Sort and get top 3
        top_apps = dict(sorted(app_traffic.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:3])
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(top_apps.keys(), 
                [val / (1024**3) for val in top_apps.values()],  # Convert to GB
                color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Top 3 Most Used Applications')
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (GB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('plots/top_3_apps.png')
        plt.close()
