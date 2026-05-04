import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import shap

# === 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
print("Загрузка данных...")
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'synthetic_health_data.csv')

df = pd.read_csv(dataset_path)

# Целевая переменная из нового датасета
target_column = 'Health_Score' 

# Отделяем признаки и таргет (ID пациента в этом датасете нет)
X = df.drop([target_column], axis=1)
y = df[target_column]

# === 2. РАЗБИЕНИЕ И ПРЕДОБРАБОТКА ===
print("Разбиение и нормализация данных...")
# Для задачи регрессии параметр stratify не используется
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)
joblib.dump(imputer, 'imputer.pkl')

# Обучаем Scaler для приведения данных к одному масштабу
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
joblib.dump(scaler, 'scaler.pkl')

# === 3. СОЗДАНИЕ И ОБУЧЕНИЕ НЕЙРОСЕТИ (РЕГРЕССИЯ) ===
print("Создание нейросети...")
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    # Для регрессии используется один выходной нейрон с линейной функцией активации
    Dense(1, activation='linear') 
])

# Компилируем модель с учетом регрессии (MSE и MAE)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("Обучение модели...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

model.save('health_score_model.keras')

# === 4. ОЦЕНКА МОДЕЛИ ===
print("\nОценка модели на тестовой выборке:")
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Предсказания для тестовой выборки
y_pred = model.predict(X_test_scaled).flatten()

print(f"MSE (Среднеквадратичная ошибка): {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score (Коэффициент детерминации): {r2_score(y_test, y_pred):.2f}")

# === 5. ИНТЕРПРЕТАЦИЯ И ГРАФИКИ (SHAP) ===
print("\nПостроение графиков SHAP (это может занять пару минут)...")
# Берем фоновую выборку для SHAP
background = shap.sample(X_train_scaled, 100)
explainer = shap.DeepExplainer(model, background)
# Ограничим тестовую выборку 100 примерами для ускорения расчетов
shap_values = explainer.shap_values(X_test_scaled[:100]) 

if isinstance(shap_values, list):
    sv = shap_values[0]
else:
    sv = shap_values

plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_test_scaled[:100], feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
print("График 'shap_summary.png' успешно сохранен в папке проекта!")