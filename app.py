from flask import Flask, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pymysql
import pickle
import os

app = Flask(__name__)

# ---------- KONFIGURASI ----------
MODEL_PATH = 'model/lstm_rssi_model.h5'
SCALER_PATH = 'model/scaler.pkl'
DB_CONFIG = {
    "host": "auth-db497.hstgr.io",
    "user": "u731251063_pgn",
    "password": "SmartMeter3",
    "database": "u731251063_pgn"
}
WINDOW_SIZE = 10


# ---------- FUNGSI UTIL ----------
def create_dataset(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][0])  # Target: kolom pertama (RSSI)
    return np.array(X), np.array(y)


# ---------- ENDPOINT: TRAINING ----------
@app.route('/train-lstm', methods=['GET'])
def train_lstm():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        query = "SELECT created_at AS timestamp, signal_strength AS rssi FROM logs ORDER BY created_at ASC"
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            return jsonify({'error': 'No data found for training'}), 400

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['rssi_smooth'] = df['rssi'].rolling(window=3, min_periods=1).mean()
        df['rssi_prev'] = df['rssi_smooth'].shift(1).fillna(method='bfill')
        df['time_seconds'] = (df.index - df.index[0]).total_seconds()

        features = ['rssi_smooth', 'rssi_prev', 'time_seconds']
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features])

        X, y = create_dataset(scaled, window_size=WINDOW_SIZE)

        train_size = int(0.8 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]

        model = Sequential()
        model.add(LSTM(64, input_shape=(WINDOW_SIZE, 3)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Simpan model dan scaler
        os.makedirs('model', exist_ok=True)
        model.save(MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)

        return jsonify({'message': 'Training berhasil! Model dan scaler disimpan.'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------- ENDPOINT: PREDIKSI RSSI ----------
@app.route('/predict-rssi', methods=['GET'])
def predict_rssi():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        query = "SELECT created_at AS timestamp, signal_strength AS rssi FROM logs ORDER BY created_at ASC"
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            return jsonify({'error': 'No data available'}), 400

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['rssi_smooth'] = df['rssi'].rolling(window=3, min_periods=1).mean()
        df['rssi_prev'] = df['rssi_smooth'].shift(1).fillna(method='bfill')
        df['time_seconds'] = (df.index - df.index[0]).total_seconds()

        features = ['rssi_smooth', 'rssi_prev', 'time_seconds']
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        scaled = scaler.transform(df[features])
        X, y_actual = create_dataset(scaled, window_size=WINDOW_SIZE)

        model = load_model(MODEL_PATH)
        y_pred = model.predict(X)

        y_actual_inv = scaler.inverse_transform(np.hstack((y_actual.reshape(-1, 1), np.zeros((len(y_actual), 2)))))[:, 0]
        y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), 2)))))[:, 0]
        timestamps = df.index[WINDOW_SIZE:].strftime('%Y-%m-%d %H:%M:%S').tolist()

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


# ---------- ENDPOINT: CEK KESEHATAN ----------
@app.route('/health', methods=['GET'])
def health_check():
    model_exists = os.path.exists(MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH)
    return jsonify({
        'status': 'healthy' if model_exists and scaler_exists else 'unhealthy',
        'model_loaded': model_exists,
        'scaler_loaded': scaler_exists
    })


# ---------- JALANKAN ----------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
