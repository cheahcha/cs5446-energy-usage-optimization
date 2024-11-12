import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def prepare_electricity_data(df):
    """
    Prepare Singapore electricity consumption data for forecasting
    
    Parameters:
    df: pandas DataFrame with date as columns and regions as index
    
    Returns:
    - df_long: Long format DataFrame with datetime index
    - df_stats: Statistical summary by region
    """
    
    df = df.copy()

    # Convert to long format
    df_long = df.reset_index().melt(
        id_vars='index',
        var_name='date',
        value_name='consumption'
    )
    df_long = df_long.rename(columns={'index': 'region'})
    
    # Extract time-based features
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long['year'] = df_long['date'].dt.year
    df_long['month'] = df_long['date'].dt.month
    df_long['quarter'] = df_long['date'].dt.quarter
    df_long['year_quarter'] = df_long['date'].dt.year.astype(str) + '-Q' + df_long['quarter'].astype(str)
    df_long['day_of_week'] = df_long['date'].dt.dayofweek
    
    # Create region categories
    region_categories = {
        'Overall': 'Overall',
        'Central Region': 'Region',
        'East Region': 'Region',
        'North East Region': 'Region',
        'North Region': 'Region',
        'West Region': 'Region'
    }
    df_long['region_category'] = df_long['region'].map(region_categories).fillna('Subzone')
    
    # Calculate consumption changes
    df_long['consumption'] = pd.to_numeric(df_long['consumption'], errors='coerce').fillna(0)
    df_long['consumption_change'] = df_long.groupby('region')['consumption'].pct_change() * 100
    df_long['consumption_yoy'] = df_long.groupby('region')['consumption'].pct_change(periods=12) * 100
    
    # Calculate rolling statistics
    df_long['consumption_ma_3m'] = df_long.groupby('region')['consumption'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df_long['consumption_ma_12m'] = df_long.groupby('region')['consumption'].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )
    
    # Calculate seasonal components
    df_long['monthly_avg'] = df_long.groupby(['region', 'month'])['consumption'].transform('mean')
    df_long['seasonal_factor'] = df_long['consumption'] / df_long['monthly_avg']
    
    # Generate summary statistics by region
    df_stats = df_long.groupby('region').agg({
        'consumption': ['count', 'mean', 'std', 'min', 'max'],
        'consumption_change': ['mean', 'std'],
        'seasonal_factor': ['mean', 'std']
    }).round(2)
    
    # Calculate peak consumption periods
    df_long['is_peak'] = df_long.groupby('region')['consumption'].transform(
        lambda x: x > x.quantile(0.75)
    )
    
    df_long = df_long[[
        'date', 'year', 'month', 'quarter', 'year_quarter',
        'region', 'region_category', 'consumption', 'consumption_change',
        'consumption_yoy', 'consumption_ma_3m', 'consumption_ma_12m',
        'monthly_avg', 'seasonal_factor', 
    ]]
    
    return df_long, df_stats

def analyze_seasonality(df_long):
    """Analyze seasonal patterns in electricity consumption"""
    seasonal_patterns = df_long.groupby(['region', 'month'])['consumption'].mean().unstack()
    peak_month = seasonal_patterns.idxmax(axis=1)
    low_month = seasonal_patterns.idxmin(axis=1)

    seasonal_patterns['seasonal_intensity'] = (seasonal_patterns.max(axis=1) - seasonal_patterns.min(axis=1)) / seasonal_patterns.mean(axis=1)
    seasonal_patterns['peak_month'] = peak_month
    seasonal_patterns['low_month'] = low_month

    return seasonal_patterns

def identify_anomalies(df_long, z_score_threshold=3):
    """Identify anomalous consumption patterns"""
    df_long['z_score'] = df_long.groupby('region')['consumption'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    anomalies = df_long[abs(df_long['z_score']) > z_score_threshold]
    return anomalies

def plot_anomalies(df_long, anomalies, regions_to_plot, save=False):

    n_regions = len(regions_to_plot)

    fig, axes = plt.subplots(n_regions, 1, figsize=(15, 5 * n_regions))
    
    if n_regions == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    for ax, region in zip(axes, regions_to_plot):
        # Filter data for the specific region
        region_data = df_long[df_long['region'] == region]
        region_anomalies = anomalies[anomalies['region'] == region]

        # Plot actual consumption data
        sns.scatterplot(data=region_data, x='date', y='consumption', ax=ax, label='Consumption', color='blue', s=50)
        
        # Highlight anomalies
        ax.scatter(region_anomalies['date'], region_anomalies['consumption'], color='red', label='Anomalies', s=100, edgecolor='black', marker='X')
        
        # Set title and labels
        ax.set_title(f'{region} Electricity Consumption with Anomalies')
        ax.set_xlabel('Date')
        ax.set_ylabel('Consumption')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig('output/anomalies.png')
    return fig

def train_forecasting_model(df_long, region_category='Subzone', forecast_periods=6):
    """
    Prepare data and create forecasts using Prophet
    
    Parameters:
    df: Original dataframe with MultiIndex columns
    region_category: Region category to predict ['Overall', 'Region', 'Subzone']
    forecast_periods: Number of months to forecast
    
    Returns:
    Dictionary containing forecasts and metrics for each region
    """

    # Filter for region category
    df = df_long.copy()
    df = df.loc[(df['region_category']==region_category) | (df['region_category']=='Overall')]

    results = {}
    
    # Prepare data for each region
    for region in df.region.unique():
        # Extract data for this region
        region_data = df.loc[df['region']==region].reset_index()
        region_data = region_data[['date', 'consumption']]
        region_data = region_data.rename(columns={'date': 'ds', 'consumption': 'y'})
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=10.0
        )
        model.add_country_holidays(country_name='SG')
        
        # Split data into train and test
        train_size = len(region_data) - 6  # Use last 6 months as test set
        train_data = region_data.iloc[:train_size]
        test_data = region_data.iloc[train_size:]
        
        model.fit(train_data)
        
        test_forecast = model.predict(test_data[['ds']])
        
        test_mae = mean_absolute_error(test_data['y'], test_forecast['yhat'])
        test_rmse = np.sqrt(root_mean_squared_error(test_data['y'], test_forecast['yhat']))
        test_r2 = r2_score(test_data['y'], test_forecast['yhat'])

        # Create a new model instance for full data training
        full_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=10.0
        )
        full_model.add_country_holidays(country_name='SG')
        
        full_model.fit(region_data)

        # Make future predictions
        future_dates = full_model.make_future_dataframe(periods=forecast_periods, freq='M')
        forecast = full_model.predict(future_dates)
        
        results[region] = {
            'model': full_model,
            'forecast': forecast,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'actual_data': region_data
        }
    
    return results



def plot_forecasts(results, regions_to_plot=['Overall'], save=False):
    """Plot forecasts for specified regions"""
    n_regions = len(regions_to_plot)
    fig, axes = plt.subplots(n_regions, 1, figsize=(15, 5*n_regions))
    if n_regions == 1:
        axes = [axes]
    
    for ax, region in zip(axes, regions_to_plot):
        result = results[region]
        forecast = result['forecast']
        actual_data = result['actual_data']
        
        # Plot actual values
        ax.plot(actual_data['ds'], actual_data['y'], 
                label='Actual', color='blue')
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 
                label='Forecast', color='red', linestyle='--')
        
        # Plot confidence interval
        ax.fill_between(forecast['ds'], 
                       forecast['yhat_lower'], 
                       forecast['yhat_upper'], 
                       color='red', alpha=0.1)
        
        ax.set_title(f'{region} Electricity Consumption Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Consumption')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()

    if save:
        plt.savefig('output/forecast.png')
    return fig

def print_metrics(results):
    
    metrics_df = pd.DataFrame({
        'Region': [],
        'MAE': [],
        'RMSE': [],
        'R2': []
    })
    
    for region, result in results.items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Region': [region],
            'MAE': [result['test_mae']],
            'RMSE': [result['test_rmse']],
            'R2': [result['test_r2']]
        })])
    
    return metrics_df.sort_values('MAE')