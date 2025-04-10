from flask import Flask, jsonify
import pandas as pd
import numpy as np
import pymysql
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from datetime import datetime
import os

app = Flask(__name__)

# 🔗 Database Config (ganti sesuai databasenya)
db_config = {
    "host": "auth-db497.hstgr.io",
    "user": "u731251063_pgn",
    "password": "SmartMeter3",
    "database": "u731251063_pgn"
}

# 📁 Path penyimpanan model
MODEL_PATH = "model/lstm_auto_model.h5"
SCALER_PATH = "model/scaler_auto.pkl"

# 🔁 Buat Dataset untuk LSTM
def create_dataset(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][0])  # target: RSSI
    return np.array(X), np.array(y)

# 🔧 Training Route
@app.route('/train-lstm', methods=['GET'])
def train_model():
    try:
        # 1️⃣ Ambil data dari DB
        conn = pymysql.connect(**db_config)
        query = "SELECT created_at AS timestamp, signal_strength AS rssi FROM logs ORDER BY created_at ASC"
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            return jsonify({'error': 'No data found in database'}), 400

        # 2️⃣ Preprocessing
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['rssi_smooth'] = df['rssi'].rolling(window=3, min_periods=1).mean()
        df['rssi_prev'] = df['rssi_smooth'].shift(1).fillna(method='bfill')
        df['time_seconds'] = (df.index - df.index[0]).total_seconds()

        features = ['rssi_smooth', 'rssi_prev', 'time_seconds']
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features])

        # 3️⃣ Buat dataset LSTM
        X, y = create_dataset(scaled, window_size=10)

        # 4️⃣ Train-test split (80-20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # 5️⃣ Build model
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(10, X.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # 6️⃣ Training
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=32,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        # 7️⃣ Save model dan scaler
        os.makedirs("model", exist_ok=True)
        model.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        # 8️⃣ Evaluasi
        y_pred = model.predict(X_test)
        y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), 2)))))[:,0]
        y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), 2)))))[:,0]

        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)

        return jsonify({
            'message': 'Model trained & saved successfully!',
            'timestamp': datetime.now().isoformat(),
            'model_path': MODEL_PATH,
            'scaler_path': SCALER_PATH,
            'metrics': {
                'mse': mse,
                'mae': mae
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 🔍 Health Check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_exists': os.path.exists(MODEL_PATH),
        'scaler_exists': os.path.exists(SCALER_PATH)
    })

if __name__ == '__main__':
    app.run(debug=True)
