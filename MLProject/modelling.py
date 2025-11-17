# Versi sederhana untuk CI Workflow

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
import os

print("Script modelling.py (CI) dimulai...")

# 1. Mendefinisikan Path Data
DATA_DIR = 'dataset_preprocessing'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_processed.csv')

# Definisikan path output untuk model
MODEL_OUTPUT_DIR = "model"
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'model.pkl')

# Buat folder 'model' jika belum ada
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# 2. Fungsi untuk Memuat Data
def load_data(train_path):
    print(f"Memuat data dari {train_path}...")
    try:
        train_df = pd.read_csv(train_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {train_path}")
        return None, None
    
    target_col = train_df.columns[-1]
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    
    print("Data berhasil dimuat.")
    return X_train, y_train

# 3. Fungsi Utama Training
def train_model():
    X_train, y_train = load_data(TRAIN_DATA_PATH)
    
    if X_train is None:
        print("Training dihentikan karena data tidak ada.")
        return

    print("Melatih model Logistic Regression...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model selesai dilatih.")
    
    # 4. Menyimpan Model ke File
    print(f"Menyimpan model ke: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print("Model berhasil disimpan.")

# 5. Menjalankan Skrip
if __name__ == "__main__":
    train_model()