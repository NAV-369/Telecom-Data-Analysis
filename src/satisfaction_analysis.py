import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# Load engagement and experience metrics (assuming they are stored in DataFrames)
# Replace these with actual loading methods as necessary
engagement_metrics = pd.read_csv('../data/engagement_metrics.csv')  # Placeholder
experience_metrics = pd.read_csv('../data/experience_metrics.csv')  # Placeholder

# Assuming clusters have been defined previously
# Replace these with actual cluster centers
less_engaged_cluster = np.array([0.2, 0.5])  # Placeholder values
worst_experience_cluster = np.array([0.8, 0.9])  # Placeholder values

# Calculate engagement scores (Euclidean distance to less engaged cluster)
engagement_scores = pairwise_distances(engagement_metrics[['Session_Frequency', 'Total_Traffic']], [less_engaged_cluster]).flatten()

# Calculate experience scores (Euclidean distance to worst experience cluster)
experience_scores = pairwise_distances(experience_metrics[['Avg_RTT', 'Avg_TCP_Retransmission']], [worst_experience_cluster]).flatten()

# Add scores to DataFrames
engagement_metrics['Engagement_Score'] = engagement_scores
experience_metrics['Experience_Score'] = experience_scores

# Combine metrics into a single DataFrame
combined_metrics = engagement_metrics.merge(experience_metrics, on='MSISDN')

# Save the combined metrics for further analysis
combined_metrics.to_csv('../data/combined_metrics.csv', index=False)

# Calculate satisfaction scores
combined_metrics['Satisfaction_Score'] = (combined_metrics['Engagement_Score'] + combined_metrics['Experience_Score']) / 2

# Get top 10 satisfied customers
top_10_satisfied = combined_metrics.nlargest(10, 'Satisfaction_Score')[['MSISDN', 'Satisfaction_Score']]

# Save the top 10 satisfied customers to a CSV file
top_10_satisfied.to_csv('../data/top_10_satisfied_customers.csv', index=False)
print('Top 10 satisfied customers saved to top_10_satisfied_customers.csv')
print('Satisfaction analysis completed and saved to combined_metrics.csv')
