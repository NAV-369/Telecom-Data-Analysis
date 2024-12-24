import pandas as pd
from experience_analytics import ExperienceAnalytics

# Load the XDR data
xdr_data_path = '../data/xdr_data.parquet'

# Load the data into a DataFrame
xdr_data = pd.read_parquet(xdr_data_path)

# Initialize the ExperienceAnalytics
experience_analyzer = ExperienceAnalytics(xdr_data)

# Calculate experience metrics
experience_metrics = experience_analyzer.task_3_2_network_metrics_analysis()

# Print the structure of the experience_metrics dictionary
print("Experience Metrics:", experience_metrics)

# Save the experience metrics to a CSV file
# Flatten the experience_metrics dictionary into a DataFrame
flattened_metrics = []
for metric_type, metrics in experience_metrics.items():
    for key, values in metrics.items():
        for value in values:
            flattened_metrics.append({'Metric_Type': metric_type, 'Key': key, 'Value': value})

experience_metrics_df = pd.DataFrame(flattened_metrics)
experience_metrics_df.to_csv('../data/experience_metrics.csv', index=False)
print('Experience metrics recalculated and saved to experience_metrics.csv')
