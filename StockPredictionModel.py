import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import streamlit as st
from utils import add_technical_indicators

def prepare_features(data, horizons=[1, 5, 20]):
    """
    Prepare features and multiple target variables for different prediction horizons.
    """
    data = add_technical_indicators(data)
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'macd', 'rsi', 'sma_20', 'sma_50', 'momentum',
        'vwap', 'atr', 'obv'
    ]
    
    # Add lagged features
    for feature in ['Close', 'Volume', 'rsi', 'macd']:
        for lag in [1, 2, 3, 5, 7, 14]:
            data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
            features.append(f'{feature}_lag_{lag}')
    
    # Rolling averages
    for window in [3, 7, 14, 30]:
        data[f'close_rolling_{window}'] = data['Close'].rolling(window=window).mean()
        features.append(f'close_rolling_{window}')
    
    # Volatility
    data['volatility_14'] = data['Close'].pct_change().rolling(window=14).std()
    features.append('volatility_14')
    
    # Temporal features
    data['day_of_week'] = data.index.dayofweek
    features.append('day_of_week')
    data['month'] = data.index.month
    features.append('month')
    
    # Multiple target variables
    targets = {}
    for horizon in horizons:
        data[f'target_{horizon}'] = data['Close'].shift(-horizon)
        targets[horizon] = f'target_{horizon}'
    
    return data, features, targets

def prepare_lstm_data(data, features, sequence_length=10):
    """
    Prepare data for LSTM models
    """
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Scale the target separately
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(data[['target']])
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_target[i+sequence_length])
    
    return np.array(X), np.array(y), scaler, target_scaler

def build_lstm_model(input_shape):
    """
    Build an LSTM model for time series prediction
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@st.cache_resource
def train_models(data, features, horizons=[1, 5, 20]):
    """
    Train multiple models for different prediction horizons.
    """
    data_with_features, features, targets = prepare_features(data, horizons)
    models = {}
    scores = {}
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    for horizon in horizons:
        # Cut off rows where target is NaN for this horizon
        horizon_data = data_with_features.iloc[:-horizon].dropna()
        X = horizon_data[features]
        y = horizon_data[targets[horizon]]
        
        models[horizon] = {}
        scores[horizon] = {}

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)
        models[horizon]['Random Forest'] = rf_model
        scores[horizon]['Random Forest'] = {
            'mae': mean_absolute_error(y, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y, rf_pred)),
            'r2': r2_score(y, rf_pred)
        }
        
        # LSTM (for longer horizons or if data is sufficient)
        if len(horizon_data) > 100:
            X_lstm, y_lstm, feature_scaler, target_scaler = prepare_lstm_data(horizon_data, features)
            X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, shuffle=False)
            lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
            lstm_pred = lstm_model.predict(X_test)
            lstm_pred = target_scaler.inverse_transform(lstm_pred)
            y_test_actual = target_scaler.inverse_transform(y_test)
            models[horizon]['LSTM'] = {
                'model': lstm_model, 'feature_scaler': feature_scaler, 'target_scaler': target_scaler, 'sequence_length': 10
            }
            scores[horizon]['LSTM'] = {
                'mae': mean_absolute_error(y_test_actual, lstm_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_actual, lstm_pred)),
                'r2': r2_score(y_test_actual.flatten(), lstm_pred.flatten())
            }
    
    return models, scores, data_with_features, features, targets

def predict_with_models(stock_data, timeframe="6M", horizons=[1, 5, 20]):
    """
    Generate consistent predictions for multiple horizons.
    """
    if len(stock_data) < 30:
        return None, None, None
    
    data_with_features = add_technical_indicators(stock_data.copy())
    data_with_features = data_with_features.replace([np.inf, -np.inf], np.nan)
    
    models, scores, prepared_data, features, targets = train_models(data_with_features, None, horizons)
    
    predictions = {}
    ensemble_predictions = {}
    confidences = {}
    
    for horizon in horizons:
        last_data = prepared_data[features].iloc[-1].values.reshape(1, -1)
        last_sequence = prepared_data[features].iloc[-10:].values
        preds = {}
        
        if 'Random Forest' in models[horizon]:
            rf_model = models[horizon]['Random Forest']
            preds['Random Forest'] = rf_model.predict(last_data)[0]
        
        if 'LSTM' in models[horizon]:
            lstm_info = models[horizon]['LSTM']
            scaled_sequence = lstm_info['feature_scaler'].transform(last_sequence)
            reshaped_sequence = np.array([scaled_sequence])
            lstm_pred = lstm_info['model'].predict(reshaped_sequence)
            preds['LSTM'] = lstm_info['target_scaler'].inverse_transform(lstm_pred)[0][0]
        
        predictions[horizon] = preds
        
        weights = {model: scores[horizon][model]['r2'] for model in preds.keys()}
        total_weight = sum(weights.values()) or 1
        ensemble_predictions[horizon] = sum(preds[model] * weights[model] / total_weight for model in preds.keys())
        confidences[horizon] = sum(scores[horizon][model]['r2'] for model in preds.keys()) / len(preds)
    
    return predictions, ensemble_predictions, confidences