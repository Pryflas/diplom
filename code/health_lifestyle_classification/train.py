import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import shap

# === 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
print("Загрузка данных...")
df = pd.read_csv('health_lifestyle_classification.csv')

# Точное название колонки с диагнозом из твоего датасета на Kaggle
target_column = 'target' 

X = df.drop([target_column, "survey_code"], axis=1)
y = df[target_column]

# Нейросеть не понимает текст в целевой переменной (диагнозе).
# Превращаем диагнозы (например, "Yes"/"No") в 1 и 0
print("Кодирование текстовых переменных...")
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
joblib.dump(target_encoder, 'target_encoder.pkl') # Сохраняем правило перевода

# Превращаем все остальные текстовые колонки (пол, привычки) в цифры (0, 1, 2...)
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

joblib.dump(label_encoders, 'label_encoders.pkl')

# === 2. РАЗБИЕНИЕ И ПРЕДОБРАБОТКА ===
print("Разбиение и нормализация данных...")
# Отделяем валидацию и тест (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Обработка пропущенных значений (NaN) - заменяем их средними значениями
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)
joblib.dump(imputer, 'imputer.pkl') # Сохраняем правило заполнения пустот

# Обучаем Scaler (нормализация чисел) ТОЛЬКО на обучающей выборке
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
joblib.dump(scaler, 'scaler.pkl') 

# Балансировка классов (создание синтетических больных пациентов для обучения)
print("Применение SMOTE (балансировка классов)...")
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# === 3. АРХИТЕКТУРА НЕЙРОСЕТИ ===
print("\nСоздание и обучение нейронной сети...")
model = Sequential([
    # Явный входной слой (40 признаков)
    Input(shape=(X_train_smote.shape[1],)),
    
    # Скрытые слои как в твоем дипломе
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    Dropout(0.3),
    
    # Выходной слой (1 нейрон для бинарной вероятности)
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC', 'accuracy'])

# Остановка обучения, если метрики на валидации перестают расти
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Запуск обучения
history = model.fit(X_train_smote, y_train_smote,
                    validation_data=(X_val_scaled, y_val),
                    epochs=100, 
                    batch_size=64, 
                    callbacks=[early_stop], 
                    verbose=1)

# Сохраняем "мозг" (готовые веса)
model.save('disease_risk_model.keras')
print("\nМодель успешно обучена и сохранена!")

# === 4. ОЦЕНКА МЕТРИК ===
print("\n=== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
y_pred_probs = model.predict(X_test_scaled)
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_probs):.4f}")
# Порог 0.5: если вероятность больше 50%, считаем пациента больным
print(classification_report(y_test, (y_pred_probs > 0.5).astype(int)))

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