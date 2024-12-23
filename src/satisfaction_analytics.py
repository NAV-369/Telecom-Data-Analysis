import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import sqlalchemy
import warnings

warnings.filterwarnings('ignore')

class SatisfactionAnalytics:
    """
    Comprehensive satisfaction analysis for telecommunication users.
    
    Implements Task 4 objectives for customer satisfaction assessment.
    """
    
    def __init__(self, engagement_data, experience_data):
        """
        Initialize SatisfactionAnalytics with engagement and experience data.
        
        Args:
            engagement_data (pd.DataFrame): User engagement metrics
            experience_data (pd.DataFrame): User experience metrics
        """
        self.engagement_data = engagement_data
        self.experience_data = experience_data
        
    def _euclidean_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1 (np.array): First data point
            point2 (np.array): Second data point
        
        Returns:
            float: Euclidean distance
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def task_4_1_compute_scores(self, engagement_clusters, experience_clusters):
        """
        Task 4.1: Compute engagement and experience scores.
        
        Args:
            engagement_clusters (dict): Clustering results from engagement analysis
            experience_clusters (dict): Clustering results from experience analysis
        
        Returns:
            pd.DataFrame: User scores
        """
        # Identify least engaged and worst experience clusters
        least_engaged_cluster = engagement_clusters['Cluster_Summary'].mean().idxmin()
        worst_experience_cluster = experience_clusters['Cluster_Summary'].mean().idxmax()
        
        # Prepare data
        engagement_features = ['Session_Frequency', 'Duration_ms', 'Total_Traffic']
        experience_features = ['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']
        
        # Compute scores
        user_scores = pd.DataFrame()
        user_scores['Customer_ID'] = self.engagement_data['Customer_ID']
        
        # Engagement Score
        engagement_scaler = StandardScaler()
        engagement_scaled = engagement_scaler.fit_transform(self.engagement_data[engagement_features])
        least_engaged_centroid = engagement_clusters['Cluster_Summary'].loc[least_engaged_cluster][engagement_features]
        least_engaged_centroid_scaled = engagement_scaler.transform(least_engaged_centroid.values.reshape(1, -1))
        
        user_scores['Engagement_Score'] = [
            self._euclidean_distance(point, least_engaged_centroid_scaled) 
            for point in engagement_scaled
        ]
        
        # Experience Score
        experience_scaler = StandardScaler()
        experience_scaled = experience_scaler.fit_transform(self.experience_data[experience_features])
        worst_experience_centroid = experience_clusters['Cluster_Summary'].loc[worst_experience_cluster][experience_features]
        worst_experience_centroid_scaled = experience_scaler.transform(worst_experience_centroid.values.reshape(1, -1))
        
        user_scores['Experience_Score'] = [
            self._euclidean_distance(point, worst_experience_centroid_scaled) 
            for point in experience_scaled
        ]
        
        return user_scores
    
    def task_4_2_satisfaction_scores(self, user_scores):
        """
        Task 4.2: Compute satisfaction scores and identify top 10 satisfied customers.
        
        Args:
            user_scores (pd.DataFrame): User engagement and experience scores
        
        Returns:
            pd.DataFrame: Top 10 satisfied customers
        """
        # Compute satisfaction score as average of engagement and experience scores
        user_scores['Satisfaction_Score'] = (user_scores['Engagement_Score'] + user_scores['Experience_Score']) / 2
        
        # Sort and return top 10 satisfied customers
        top_10_satisfied = user_scores.nlargest(10, 'Satisfaction_Score')
        
        return top_10_satisfied
    
    def task_4_3_satisfaction_prediction(self, user_scores):
        """
        Task 4.3: Build regression model to predict satisfaction score.
        
        Args:
            user_scores (pd.DataFrame): User scores
        
        Returns:
            dict: Regression model performance metrics
        """
        # Prepare features and target
        X = user_scores[['Engagement_Score', 'Experience_Score']]
        y = user_scores['Satisfaction_Score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        return {
            'Model': model,
            'MSE': mean_squared_error(y_test, y_pred),
            'R2_Score': r2_score(y_test, y_pred)
        }
    
    def task_4_4_4_5_clustering_analysis(self, user_scores):
        """
        Task 4.4 & 4.5: Perform K-means clustering and aggregate results.
        
        Args:
            user_scores (pd.DataFrame): User scores
        
        Returns:
            dict: Clustering results and aggregated metrics
        """
        # Prepare data for clustering
        X = user_scores[['Engagement_Score', 'Experience_Score']]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        user_scores['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Aggregate metrics per cluster
        cluster_metrics = user_scores.groupby('Cluster').agg({
            'Engagement_Score': 'mean',
            'Experience_Score': 'mean',
            'Satisfaction_Score': 'mean'
        })
        
        return {
            'Cluster_Metrics': cluster_metrics,
            'Clustered_Data': user_scores
        }
    
    def task_4_6_export_to_mysql(self, user_scores, db_config):
        """
        Task 4.6: Export user scores to MySQL database.
        
        Args:
            user_scores (pd.DataFrame): User scores
            db_config (dict): Database configuration
        
        Returns:
            bool: Export success status
        """
        try:
            # Create SQLAlchemy engine
            engine = sqlalchemy.create_engine(
                f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
            )
            
            # Export to MySQL
            user_scores.to_sql(
                name='user_satisfaction_scores', 
                con=engine, 
                if_exists='replace', 
                index=False
            )
            
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def run_complete_analysis(self, engagement_clusters, experience_clusters, db_config=None):
        """
        Run complete satisfaction analysis pipeline.
        
        Args:
            engagement_clusters (dict): Engagement clustering results
            experience_clusters (dict): Experience clustering results
            db_config (dict, optional): MySQL database configuration
        
        Returns:
            dict: Comprehensive analysis results
        """
        # Compute user scores
        user_scores = self.task_4_1_compute_scores(engagement_clusters, experience_clusters)
        
        # Identify top satisfied customers
        top_10_satisfied = self.task_4_2_satisfaction_scores(user_scores)
        
        # Build prediction model
        prediction_model = self.task_4_3_satisfaction_prediction(user_scores)
        
        # Perform clustering analysis
        clustering_results = self.task_4_4_4_5_clustering_analysis(user_scores)
        
        # Export to MySQL if configuration provided
        if db_config:
            export_status = self.task_4_6_export_to_mysql(user_scores, db_config)
        else:
            export_status = False
        
        return {
            'User_Scores': user_scores,
            'Top_10_Satisfied': top_10_satisfied,
            'Prediction_Model': prediction_model,
            'Clustering_Results': clustering_results,
            'MySQL_Export_Status': export_status
        }
