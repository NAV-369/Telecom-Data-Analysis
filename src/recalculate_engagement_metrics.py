import pandas as pd
from user_engagement import UserEngagementAnalyzer

# Load the XDR data
xdr_data_path = '../data/xdr_data.parquet'

# Load the data into a DataFrame
xdr_data = pd.read_parquet(xdr_data_path)

# Initialize the UserEngagementAnalyzer
engagement_analyzer = UserEngagementAnalyzer(xdr_data)

# Calculate engagement metrics
engagement_metrics = engagement_analyzer.aggregate_engagement_metrics()

# Save the engagement metrics to a CSV file
engagement_metrics.to_csv('../data/engagement_metrics.csv', index=False)
print('Engagement metrics recalculated and saved to engagement_metrics.csv')
