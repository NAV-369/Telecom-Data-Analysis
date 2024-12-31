import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import DataPreparation
from user_engagement import UserEngagementAnalyzer

def main():
    """Run user engagement analysis."""
    # Load and prepare data
    data_prep = DataPreparation()
    data_prep.load_data()
    
    # Initialize engagement analyzer
    engagement_analyzer = UserEngagementAnalyzer(data_prep.xdr_data)
    
    # 1. Aggregate engagement metrics and get top 10 users
    print("\nTop 10 Users by Different Metrics:")
    metrics = ['Session_Frequency', 'Duration_ms', 'Total_Traffic']
    for metric in metrics:
        print(f"\nTop 10 Users by {metric}:")
        top_users = engagement_analyzer.get_top_users(metric)
        print(top_users[['MSISDN', metric]])
    
    # 2. Perform k-means clustering
    print("\nPerforming K-means Clustering (k=3)...")
    clustered_data, cluster_stats = engagement_analyzer.perform_kmeans(k=3)
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    for cluster, stats in cluster_stats.items():
        print(f"\n{cluster}:")
        print(f"Size: {stats['size']}")
        print("Average metrics:")
        print(stats['mean'])
    
    # 3. Find optimal k using elbow method
    print("\nFinding optimal k...")
    k_values, inertias, silhouette_scores = engagement_analyzer.find_optimal_k()
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    # Inertia plot
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    # Silhouette score plot
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.savefig('plots/clustering_analysis.png')
    plt.close()
    
    # 4. Analyze application engagement
    print("\nAnalyzing application engagement...")
    app_engagement = engagement_analyzer.analyze_app_engagement()
    print("\nTop users per application:")
    print(app_engagement)
    
    # 5. Plot top 3 applications
    print("\nPlotting top 3 applications...")
    engagement_analyzer.plot_top_apps()
    
    print("\nAnalysis complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main()
