import pytest
import pandas as pd
import numpy as np
from src.user_engagement import UserEngagementAnalyzer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'MSISDN/Number': ['user1', 'user1', 'user2', 'user2', 'user3'],
        'Session ID': [1, 2, 1, 2, 1],
        'Dur. (ms)': [100, 200, 150, 250, 300],
        'Total DL (Bytes)': ['1000', '2000', '1500', '2500', '3000'],
        'Total UL (Bytes)': ['500', '1000', '750', '1250', '1500'],
        'Social Media DL (Bytes)': ['100', '200', '150', '250', '300'],
        'Social Media UL (Bytes)': ['50', '100', '75', '125', '150']
    }
    return pd.DataFrame(data)

def test_aggregate_engagement_metrics(sample_data):
    """Test engagement metrics aggregation."""
    analyzer = UserEngagementAnalyzer(sample_data)
    metrics = analyzer.aggregate_engagement_metrics()
    
    assert len(metrics) == 3  # Three unique users
    assert 'Session_Frequency' in metrics.columns
    assert 'Duration_ms' in metrics.columns
    assert 'Total_Traffic' in metrics.columns
    
    # Check user1's metrics
    user1_metrics = metrics[metrics['MSISDN'] == 'user1'].iloc[0]
    assert user1_metrics['Session_Frequency'] == 2
    assert user1_metrics['Duration_ms'] == 300
    assert user1_metrics['Total_Traffic'] == 4500

def test_normalize_metrics(sample_data):
    """Test metrics normalization."""
    analyzer = UserEngagementAnalyzer(sample_data)
    normalized = analyzer.normalize_metrics()
    
    # Check if normalized data has mean close to 0 and std close to 1
    metrics = ['Session_Frequency', 'Duration_ms', 'Total_Traffic']
    for metric in metrics:
        assert abs(normalized[metric].mean()) < 1e-10
        assert abs(normalized[metric].std() - 1) < 1e-10

def test_perform_kmeans(sample_data):
    """Test k-means clustering."""
    analyzer = UserEngagementAnalyzer(sample_data)
    clustered_data, stats = analyzer.perform_kmeans(k=2)
    
    assert 'Cluster' in clustered_data.columns
    assert len(stats) == 2  # Two clusters
    assert all(cluster in stats for cluster in ['Cluster_0', 'Cluster_1'])

def test_find_optimal_k(sample_data):
    """Test optimal k finding."""
    analyzer = UserEngagementAnalyzer(sample_data)
    k_values, inertias, silhouette_scores = analyzer.find_optimal_k(max_k=3)
    
    assert len(k_values) == 2  # k from 2 to 3
    assert len(inertias) == 2
    assert len(silhouette_scores) == 2
    assert all(i > 0 for i in inertias)
    assert all(-1 <= s <= 1 for s in silhouette_scores)

def test_analyze_app_engagement(sample_data):
    """Test application engagement analysis."""
    analyzer = UserEngagementAnalyzer(sample_data)
    app_engagement = analyzer.analyze_app_engagement()
    
    assert not app_engagement.empty
    assert 'Social Media DL (Bytes)_Top_Users' in app_engagement.columns
    assert 'Social Media UL (Bytes)_Top_Users' in app_engagement.columns
