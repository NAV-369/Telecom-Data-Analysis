# Telecom Data Analysis

A comprehensive telecom data analysis project with an interactive dashboard for visualizing user behavior, handset analysis, and network performance metrics.

## Features

- Interactive dashboard using Streamlit
- User behavior analysis
- Handset and manufacturer analysis
- Network performance metrics
- Advanced analytics with PCA
- Time series analysis of data usage

## Project Structure

```
.
├── src/               # Source code
├── tests/             # Unit tests
├── notebooks/         # Jupyter notebooks
├── data/             # Data directory (not tracked in git)
├── plots/            # Generated visualizations
└── report/           # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/NAV-369/Telecom-Data-Analysis.git
cd Telecom-Data-Analysis
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

The project requires two main data files that are not included in the repository due to their size:
- `data/telecom.sql`: SQL dump file containing the telecom database
- `data/xdr_data.parquet`: Parquet file containing the XDR data

To set up the data:
1. Create a `data` directory in the project root if it doesn't exist
2. Place the required data files in the `data` directory
3. The application will automatically detect and use these files

## Running the Dashboard

To run the interactive dashboard:
```bash
streamlit run src/dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## Features

### User Behavior Analysis
- Data usage patterns
- Session analysis
- Application usage breakdown
- User segmentation

### Handset Analysis
- Top handset models
- Manufacturer market share
- Model-specific metrics

### Network Performance
- Throughput analysis
- RTT measurements
- TCP retransmission analysis

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

## License

[MIT License](LICENSE)
