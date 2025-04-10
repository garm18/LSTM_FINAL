import unittest
import requests

class TestLSTMTrainAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:5000"  # ganti dengan IP VPS jika di-deploy

    def test_health_check(self):
        """Cek apakah API dalam kondisi sehat"""
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('model_loaded', data)
        self.assertIn('scaler_loaded', data)
        print("[✅] /health endpoint OK")

    def test_train_lstm(self):
        """Test endpoint training LSTM"""
        response = requests.get(f"{self.BASE_URL}/train-lstm")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validasi struktur response
        self.assertIn("message", data)
        self.assertIn("timestamp", data)
        self.assertIn("model_path", data)
        self.assertIn("scaler_path", data)
        self.assertIn("metrics", data)

        # Validasi isi metrics
        metrics = data["metrics"]
        self.assertIn("mse", metrics)
        self.assertIn("mae", metrics)
        self.assertIsInstance(metrics["mse"], float)
        self.assertIsInstance(metrics["mae"], float)
        print("[✅] /train-lstm endpoint berhasil menghasilkan model")

if __name__ == "__main__":
    unittest.main()
