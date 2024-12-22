import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    """Class for data preparation and cleaning for telecom data analysis."""
    
    def __init__(self):
        """Initialize the DataPreparation class."""
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.xdr_data = None
        self.pipeline = None
        
    def load_data(self) -> None:
        """Load data from parquet file."""
        try:
            # Load data from parquet file
            xdr_file = self.data_dir / 'xdr_data.parquet'
            self.xdr_data = pd.read_parquet(xdr_file)
            logger.info("Data loaded successfully")
            
            # Convert columns to appropriate types
            numeric_columns = [
                'Bearer Id', 'Start ms', 'End ms', 'Dur. (ms)', 'IMSI',
                'MSISDN/Number', 'IMEI', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
                'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)',
                '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
                'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)',
                '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
                'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
                'Activity Duration UL (ms)', 'Dur. (ms).1'
            ]
            
            for col in numeric_columns:
                self.xdr_data[col] = pd.to_numeric(self.xdr_data[col], errors='coerce')
            
            # Create derived columns
            self.xdr_data['session_id'] = range(len(self.xdr_data))
            self.xdr_data['duration'] = self.xdr_data['Dur. (ms)']
            
            # Handle NULL values and convert to float
            self.xdr_data['download_data'] = pd.to_numeric(
                self.xdr_data['Total DL (Bytes)'].replace('\\N', np.nan), 
                errors='coerce'
            )
            self.xdr_data['upload_data'] = pd.to_numeric(
                self.xdr_data['Total UL (Bytes)'].replace('\\N', np.nan), 
                errors='coerce'
            )
            self.xdr_data['msisdn'] = self.xdr_data['MSISDN/Number']
            
            # Create application-specific columns
            apps = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']
            for app in apps:
                dl_col = f'{app} DL (Bytes)'
                ul_col = f'{app} UL (Bytes)'
                total_col = f'total_data_volume_{app.lower()}'
                
                # Handle NULL values and convert to float
                dl_data = pd.to_numeric(
                    self.xdr_data[dl_col].replace('\\N', np.nan), 
                    errors='coerce'
                )
                ul_data = pd.to_numeric(
                    self.xdr_data[ul_col].replace('\\N', np.nan), 
                    errors='coerce'
                )
                self.xdr_data[total_col] = dl_data + ul_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def analyze_handsets(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Analyze handset data.
        
        Returns:
            Tuple containing:
            - Top 10 handsets DataFrame
            - Top 3 manufacturers DataFrame
            - Dictionary with top 5 handsets per manufacturer
        """
        # Top 10 handsets
        top_10_handsets = self.xdr_data['Handset Type'].value_counts().head(10).reset_index()
        top_10_handsets.columns = ['handset', 'count']
        
        # Top 3 manufacturers
        top_3_manufacturers = self.xdr_data['Handset Manufacturer'].value_counts().head(3).reset_index()
        top_3_manufacturers.columns = ['manufacturer', 'count']
        
        # Top 5 handsets per manufacturer
        top_5_per_manufacturer = {}
        for manufacturer in top_3_manufacturers['manufacturer']:
            manufacturer_handsets = self.xdr_data[self.xdr_data['Handset Manufacturer'] == manufacturer]
            top_5 = manufacturer_handsets['Handset Type'].value_counts().head(5).reset_index()
            top_5.columns = ['handset', 'count']
            top_5_per_manufacturer[manufacturer] = top_5
            
        return top_10_handsets, top_3_manufacturers, top_5_per_manufacturer
    
    def aggregate_user_behavior(self) -> pd.DataFrame:
        """Aggregate user behavior data.
        
        Returns:
            DataFrame with aggregated user metrics
        """
        user_metrics = self.xdr_data.groupby('msisdn').agg({
            'session_id': 'count',  # number of xDR sessions
            'duration': 'sum',      # total session duration
            'download_data': 'sum', # total download
            'upload_data': 'sum',   # total upload
        }).reset_index()
        
        # Calculate total data volume
        user_metrics['total_data_volume'] = user_metrics['download_data'] + user_metrics['upload_data']
        
        # Add application-specific metrics
        apps = ['social media', 'google', 'email', 'youtube', 'netflix', 'gaming', 'other']
        for app in apps:
            app_col = f'total_data_volume_{app}'
            app_metrics = self.xdr_data.groupby('msisdn')[app_col].sum().reset_index()
            user_metrics = user_metrics.merge(app_metrics, on='msisdn', how='left')
        
        return user_metrics
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        # Replace missing values with mean for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        # Replace missing values with mode for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers in specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
            method: Method to handle outliers ('iqr' or 'zscore')
            
        Returns:
            DataFrame with handled outliers
        """
        df_clean = df.copy()
        
        for column in columns:
            # Ensure column is numeric
            df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
            
            if method == 'iqr':
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
                df_clean[column] = df_clean[column].mask(z_scores > 3, df_clean[column].mean())
                
        return df_clean
    
    def create_deciles(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Create decile classes based on a column.
        
        Args:
            df: Input DataFrame
            column: Column to base deciles on
            
        Returns:
            DataFrame with decile classes
        """
        df = df.copy()
        df['decile_class'] = pd.qcut(df[column], q=10, labels=['D1', 'D2', 'D3', 'D4', 'D5', 
                                                              'D6', 'D7', 'D8', 'D9', 'D10'])
        return df
    
    def compute_correlation_matrix(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Compute correlation matrix for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to include in correlation matrix
            
        Returns:
            Correlation matrix DataFrame
        """
        return df[columns].corr()
    
    def perform_pca(self, df: pd.DataFrame, columns: List[str], n_components: int = 2) -> Tuple[pd.DataFrame, float]:
        """Perform PCA on specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to include in PCA
            n_components: Number of components to keep
            
        Returns:
            Tuple containing:
            - DataFrame with PCA results
            - Explained variance ratio
        """
        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[columns])
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return pca_df, pca.explained_variance_ratio_
