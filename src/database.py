from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create base class for declarative models
Base = declarative_base()

class UserMetrics(Base):
    """Table for storing user metrics"""
    __tablename__ = 'user_metrics'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True)
    xdr_sessions = Column(Integer)
    total_duration = Column(Float)
    total_download = Column(Float)
    total_upload = Column(Float)
    total_data_volume = Column(Float)

class ApplicationMetrics(Base):
    """Table for storing application-specific metrics"""
    __tablename__ = 'application_metrics'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    app_name = Column(String)
    session_count = Column(Integer)
    total_duration = Column(Float)
    data_volume = Column(Float)

def get_database_connection():
    """Create database connection"""
    db_url = os.getenv('DATABASE_URL', 'sqlite:///telecom_analysis.db')
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def store_user_metrics(session, metrics_df):
    """Store user metrics in database"""
    for _, row in metrics_df.iterrows():
        metric = UserMetrics(
            user_id=row['user_id'],
            xdr_sessions=row['session_count'],
            total_duration=row['total_duration'],
            total_download=row['total_download'],
            total_upload=row['total_upload'],
            total_data_volume=row['total_data_volume']
        )
        session.add(metric)
    session.commit()

def store_app_metrics(session, app_metrics_df):
    """Store application metrics in database"""
    for _, row in app_metrics_df.iterrows():
        metric = ApplicationMetrics(
            user_id=row['user_id'],
            app_name=row['app_name'],
            session_count=row['session_count'],
            total_duration=row['duration'],
            data_volume=row['data_volume']
        )
        session.add(metric)
    session.commit()
