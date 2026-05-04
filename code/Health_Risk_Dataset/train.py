import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import shap

# === 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
print("Загрузка данных...")
current_dir = os.path.dirname(os.path.abspath(__file__))
# Склеиваем путь к папке с именем файла
dataset_path = os.path.join(current_dir, 'Health_Risk_Dataset.csv')

# Читаем файл по абсолютному пути
df = pd.read_csv(dataset_path)

# Целевая переменная из датасета
target_column = 'Risk_Level' 

# Удаляем целевую переменную и ID пациента (он не нужен для предсказаний)
X = df.drop([target_column, 'Patient_ID'], axis=1)
y = df[target_column]

# Кодируем 4 класса риска (Low, Normal, Medium, High) в числа (0, 1, 2, 3)
print("Кодирование целевой переменной...")
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
num_classes = len(target_encoder.classes_)
joblib.dump(target_encoder, 'target_encoder.pkl') 

# Превращаем категориальные колонки (например, Consciousness) в цифры
print("Кодирование текстовых признаков...")
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le
joblib.dump(label_encoders, 'label_encoders.pkl')

# === 2. РАЗБИЕНИЕ И ПРЕДОБРАБОТКА ===
print("Разбиение и нормализация данных...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)
joblib.dump(imputer, 'imputer.pkl') 

# Обучаем Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
joblib.dump(scaler, 'scaler.pkl') 

# Балансировка классов
print("Применение SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# === 3. АРХИТЕКТУРА НЕЙРОСЕТИ ===
print("\nСоздание и обучение нейронной сети...")
model = Sequential([
    Input(shape=(X_train_smote.shape[1],)),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    Dropout(0.3),
    
    # Для многоклассовой классификации используем softmax и количество нейронов = num_classes
    Dense(num_classes, activation='softmax')
])

# Изменена функция потерь на sparse_categorical_crossentropy
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_smote, y_train_smote,
                    validation_data=(X_val_scaled, y_val),
                    epochs=100, 
                    batch_size=64, 
                    callbacks=[early_stop], 
                    verbose=1)

model.save('disease_risk_model.keras')
print("\nМодель успешно обучена и сохранена!")

# === 4. ОЦЕНКА МЕТРИК ===
print("\n=== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
y_pred_probs = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_probs, multi_class='ovr'):.4f}")
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred_classes, target_names=target_encoder.classes_))

# # === 5. ИНТЕРПРЕТАЦИЯ И ГРАФИКИ (SHAP) ===
# print("\nПостроение графиков SHAP...")
# background = shap.sample(X_train_smote, 100)
# explainer = shap.DeepExplainer(model, background)
# shap_values = explainer.shap_values(X_test_scaled)
# plt.figure(figsize=(10, 6))
# # shap_values для softmax обычно является списком массивов (по одному на каждый класс)
# shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, class_names=list(target_encoder.classes_), show=False)
# plt.tight_layout()
# plt.savefig('shap_summary.png')
# print("График 'shap_summary.png' успешно сохранен в папке проекта!")

# === 5. ИНТЕРПРЕТАЦИЯ И ГРАФИКИ (SHAP) ===
print("\nПостроение графиков SHAP (это может занять пару минут)...")
background = shap.sample(X_train_smote, 100)
explainer = shap.DeepExplainer(model, background)

shap_values = explainer.shap_values(X_test_scaled)

# ИСПРАВЛЕНИЕ: Приводим размерности в порядок для новых версий Keras
if isinstance(shap_values, list):
    sv = shap_values[0] # Берем первый элемент, если это список
else:
    sv = shap_values

# Если массив 3-мерный (например, 15000, 47, 1), отбрасываем последнее измерение
if len(sv.shape) == 3:
    sv = sv[:, :, 0]

plt.figure(figsize=(10, 6))
# Передаем очищенный массив sv
shap.summary_plot(sv, X_test_scaled, feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
print("График 'shap_summary.png' успешно сохранен в папке проекта!")