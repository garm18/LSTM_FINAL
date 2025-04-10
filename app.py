from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pymysql
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os

app = Flask(__name__)

# Konfigurasi
db_config = {
    "host": "auth-db497.hstgr.io",
    "user": "u731251063_pgn",
    "password": "SmartMeter3",
    "database": "u731251063_pgn"
}
MODEL_PATH = "model/lstm_rssi_model.h5"
SCALER_PATH = "model/scaler.pkl"

# Cek model & scaler
model = load_model(MODEL_PATH, compile=False) if os.path.exists(MODEL_PATH) else None
scaler = pickle.load(open(SCALER_PATH, "rb")) if os.path.exists(SCALER_PATH) else None

def create_dataset(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][0])
    return np.array(X), np.array(y)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

@app.route('/train-lstm', methods=['GET'])
def train_lstm():
    try:
        conn = pymysql.connect(**db_config)
        query = "SELECT created_at AS timestamp, signal_strength AS rssi FROM logs ORDER BY created_at ASC"
        df = pd.read_sql(query, conn)
        conn.close()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['rssi_smooth'] = df['rssi'].rolling(window=3, min_periods=1).mean()
        df['rssi_prev'] = df['rssi_smooth'].shift(1).fillna(method='bfill')
        df['time_seconds'] = (df.index - df.index[0]).total_seconds()

        features = ['rssi_smooth', 'rssi_prev', 'time_seconds']
        scaler_new = MinMaxScaler()
        scaled = scaler_new.fit_transform(df[features])

        X, y = create_dataset(scaled, window_size=10)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_new = Sequential()
        model_new.add(LSTM(64, activation='tanh', input_shape=(10, 3)))
        model_new.add(Dense(1))
        model_new.compile(optimizer='adam', loss='mse')

        model_new.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[EarlyStopping(patience=3)], verbose=0)

        model_new.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler_new, f)

        return jsonify({"message": "Model retrained successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-rssi', methods=['GET'])
def predict_rssi():
    try:
        conn = pymysql.connect(**db_config)
        query = "SELECT created_at AS timestamp, signal_strength AS rssi FROM logs ORDER BY created_at ASC"
        df = pd.read_sql(query, conn)
        conn.close()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['rssi_smooth'] = df['rssi'].rolling(window=3, min_periods=1).mean()
        df['rssi_prev'] = df['rssi_smooth'].shift(1).fillna(method='bfill')
        df['time_seconds'] = (df.index - df.index[0]).total_seconds()

        features = ['rssi_smooth', 'rssi_prev', 'time_seconds']
        scaler_loaded = pickle.load(open(SCALER_PATH, 'rb'))
        scaled = scaler_loaded.transform(df[features])

        X, y_actual = create_dataset(scaled, window_size=10)
        model_loaded = tf.keras.models.load_model(MODEL_PATH, compile=False)
        y_pred = model_loaded.predict(X)

        y_actual_inv = scaler_loaded.inverse_transform(np.hstack((y_actual.reshape(-1, 1), np.zeros((len(y_actual), 2)))))[:, 0]
        y_pred_inv = scaler_loaded.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), 2)))))[:, 0]
        timestamps = df.index[10:].strftime('%Y-%m-%d %H:%M:%S').tolist()

        results = []
        for t, a, p in zip(timestamps, y_actual_inv, y_pred_inv):
            results.append({
                'timestamp': t,
                'actual_rssi': round(float(a), 2),
                'predicted_rssi': round(float(p), 2)
            })

        return jsonify({'data': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
