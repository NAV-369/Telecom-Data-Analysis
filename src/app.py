import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import get_database_connection

def load_data():
    """Load data from database"""
    session = get_database_connection()
    user_metrics = pd.read_sql('SELECT * FROM user_metrics', session.bind)
    app_metrics = pd.read_sql('SELECT * FROM application_metrics', session.bind)
    return user_metrics, app_metrics

def main():
    st.title('Telecom User Analysis Dashboard')
    
    # Sidebar
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select a page:', 
                           ['Overview', 'User Behavior', 'Application Analysis'])
    
    # Load data
    user_metrics, app_metrics = load_data()
    
    if page == 'Overview':
        st.header('User Overview')
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Users', len(user_metrics['user_id'].unique()))
        with col2:
            st.metric('Total Sessions', user_metrics['xdr_sessions'].sum())
        with col3:
            st.metric('Total Data Volume (GB)', 
                     round(user_metrics['total_data_volume'].sum() / 1e9, 2))
        
        # Data volume distribution
        st.subheader('Data Volume Distribution')
        fig = px.histogram(user_metrics, x='total_data_volume',
                          title='Distribution of Total Data Volume per User')
        st.plotly_chart(fig)
        
    elif page == 'User Behavior':
        st.header('User Behavior Analysis')
        
        # Session duration vs Data volume
        st.subheader('Session Duration vs Data Volume')
        fig = px.scatter(user_metrics, 
                        x='total_duration', 
                        y='total_data_volume',
                        title='Session Duration vs Data Volume')
        st.plotly_chart(fig)
        
        # Top users by data volume
        st.subheader('Top Users by Data Volume')
        top_users = user_metrics.nlargest(10, 'total_data_volume')
        fig = px.bar(top_users, 
                    x='user_id', 
                    y='total_data_volume',
                    title='Top 10 Users by Data Volume')
        st.plotly_chart(fig)
        
    else:  # Application Analysis
        st.header('Application Analysis')
        
        # Application usage distribution
        st.subheader('Application Usage Distribution')
        app_usage = app_metrics.groupby('app_name')['data_volume'].sum()
        fig = px.pie(values=app_usage.values, 
                    names=app_usage.index,
                    title='Data Volume Distribution by Application')
        st.plotly_chart(fig)
        
        # Application session counts
        st.subheader('Application Session Counts')
        app_sessions = app_metrics.groupby('app_name')['session_count'].sum()
        fig = px.bar(x=app_sessions.index, 
                    y=app_sessions.values,
                    title='Number of Sessions by Application')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
