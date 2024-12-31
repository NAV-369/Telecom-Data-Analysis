# Dashboard Setup and Deployment Instructions

## Prerequisites
- Python 3.8+
- Git
- pip (Python package manager)

## Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/NAV-369/Telecom-Data-Analysis.git
cd Telecom-Data-Analysis
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Data Preparation
- Ensure `data/xdr_data.parquet` exists in the project directory
- If not, contact project maintainers for data file

### 5. Run Dashboard
```bash
streamlit run src/dashboard.py
```

## Troubleshooting
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify data file location
- Install additional system dependencies if required

## Deployment Options
1. **Local Development**: Follow setup instructions above
2. **Cloud Deployment**: 
   - Streamlit Cloud
   - Heroku
   - AWS Elastic Beanstalk

## Contact
For issues, contact project maintainers at [your-email@example.com]
