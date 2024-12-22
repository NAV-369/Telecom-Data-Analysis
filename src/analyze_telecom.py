from data_preparation import DataPreparation
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the telecom data analysis."""
    try:
        # Initialize data preparation
        data_prep = DataPreparation()
        
        # Load data
        logger.info("Loading data...")
        data_prep.load_data()
        
        # Analyze handsets
        logger.info("Analyzing handsets...")
        top_10_handsets, top_3_manufacturers, top_5_per_manufacturer = data_prep.analyze_handsets()
        
        # Create plots directory if it doesn't exist
        plots_dir = Path(__file__).parent.parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot top 10 handsets
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_10_handsets, x='handset', y='count')
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Handsets')
        plt.tight_layout()
        plt.savefig(plots_dir / 'top_10_handsets.png')
        plt.close()
        
        # Plot top 3 manufacturers
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_3_manufacturers, x='manufacturer', y='count')
        plt.title('Top 3 Manufacturers')
        plt.tight_layout()
        plt.savefig(plots_dir / 'top_3_manufacturers.png')
        plt.close()
        
        # Plot top 5 handsets for each manufacturer
        for manufacturer, top_5 in top_5_per_manufacturer.items():
            plt.figure(figsize=(12, 6))
            sns.barplot(data=top_5, x='handset', y='count')
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Top 5 Handsets - {manufacturer}')
            plt.tight_layout()
            plt.savefig(plots_dir / f'top_5_handsets_{manufacturer}.png')
            plt.close()
        
        # Analyze user behavior
        logger.info("Analyzing user behavior...")
        user_metrics = data_prep.aggregate_user_behavior()
        
        # Handle missing values and outliers
        user_metrics = data_prep.handle_missing_values(user_metrics)
        user_metrics = data_prep.handle_outliers(
            user_metrics, 
            ['total_data_volume', 'duration', 'session_id']
        )
        
        # Create deciles based on total data volume
        user_metrics = data_prep.create_deciles(user_metrics, 'total_data_volume')
        
        # Plot user behavior metrics
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=user_metrics, x='decile_class', y='total_data_volume')
        plt.title('Total Data Volume by Decile')
        plt.tight_layout()
        plt.savefig(plots_dir / 'data_volume_deciles.png')
        plt.close()
        
        # Compute correlation matrix for numeric columns
        numeric_columns = ['session_id', 'duration', 'download_data', 'upload_data', 'total_data_volume']
        correlation_matrix = data_prep.compute_correlation_matrix(user_metrics, numeric_columns)
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_matrix.png')
        plt.close()
        
        # Perform PCA
        pca_columns = ['session_id', 'duration', 'download_data', 'upload_data']
        pca_results, explained_variance = data_prep.perform_pca(user_metrics, pca_columns)
        
        # Plot PCA results
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_results['PC1'], pca_results['PC2'], alpha=0.5)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance explained)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance explained)')
        plt.title('PCA Results')
        plt.tight_layout()
        plt.savefig(plots_dir / 'pca_results.png')
        plt.close()
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
