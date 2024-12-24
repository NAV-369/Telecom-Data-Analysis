import pandas as pd

# Load engagement metrics
engagement_metrics = pd.read_csv('../data/engagement_metrics.csv')

# Load experience metrics
experience_metrics = pd.read_csv('../data/experience_metrics.csv')

# Assuming both DataFrames have a common key to merge on, e.g., 'MSISDN'
combined_metrics = pd.merge(engagement_metrics, experience_metrics, on='MSISDN', how='inner')

# Save the combined metrics to a CSV file
combined_metrics.to_csv('../data/combined_metrics.csv', index=False)
print('Combined metrics saved to combined_metrics.csv')
