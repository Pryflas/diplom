import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
import shap

# === 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
print("Загрузка данных...")
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'diabetes_binary_health_indicators_BRFSS2015.csv')

df = pd.read_csv(dataset_path)

target_column = 'Diabetes_binary' 
X = df.drop([target_column], axis=1)
y = df[target_column]

print("Кодирование целевой переменной...")
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
joblib.dump(target_encoder, 'target_encoder.pkl') 

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

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)
joblib.dump(imputer, 'imputer.pkl') 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
joblib.dump(scaler, 'scaler.pkl') 

print("Вычисление весов классов...")
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(weights))
print(f"Веса классов: {class_weights_dict}")

# === 3. АВТОМАТИЧЕСКИЙ ПОДБОР ПАРАМЕТРОВ (KERAS TUNER) ===
print("\nНастройка функции создания модели для KerasTuner...")

def build_model(hp):
    """Функция для сборки модели с варьируемыми гиперпараметрами"""
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    
    # Перебираем количество скрытых слоев от 1 до 4
    for i in range(hp.Int('num_layers', 1, 4)):
        # Подбираем количество нейронов
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
        
        # Подбираем, нужен ли BatchNormalization
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(BatchNormalization())
            
        # Подбираем уровень Dropout
        model.add(Dropout(
            rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        ))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Подбираем скорость обучения (Learning Rate)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model

# Используем алгоритм Hyperband для быстрого поиска лучших параметров
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_AUC", direction="max"), # Ищем модель с максимальным валидационным AUC
    max_epochs=50,
    factor=3,
    directory='tuner_logs',
    project_name='diabetes_risk_tuning',
    overwrite=True
)

early_stop = EarlyStopping(monitor='val_AUC', mode='max', patience=10, restore_best_weights=True)

print("Начало поиска лучших гиперпараметров...")
tuner.search(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=512,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)

# === 4. ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ ===
print("\n=== ЛУЧШИЕ ПАРАМЕТРЫ НАЙДЕНЫ ===")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Количество слоев: {best_hps.get('num_layers')}")
print(f"Скорость обучения: {best_hps.get('learning_rate')}")

# Строим модель с лучшими параметрами
best_model = tuner.hypermodel.build(best_hps)

# Обучаем её до конца (Tuner только ищет, теперь обучаем начисто)
reduce_lr = ReduceLROnPlateau(monitor='val_AUC', mode='max', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

print("\nФинальное обучение лучшей модели...")
history = best_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100, 
    batch_size=512, 
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

best_model.save('best_disease_risk_model.keras')
print("\nЛучшая модель успешно обучена и сохранена!")

# === 5. ОЦЕНКА МЕТРИК НА ТЕСТОВОЙ ВЫБОРКЕ ===
print("\n=== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
y_pred_probs = best_model.predict(X_test_scaled)

auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"ROC-AUC: {auc_score:.4f}")

y_pred_classes = (y_pred_probs > 0.5).astype(int)
print("\nОтчет о классификации (Порог 0.5):")
print(classification_report(y_test, y_pred_classes, target_names=['Нет диабета', 'Диабет']))

# === 6. ИНТЕРПРЕТАЦИЯ И ГРАФИКИ (SHAP) ===
print("\nПостроение графиков SHAP (это может занять пару минут)...")
# ИСПРАВЛЕНО: Заменен X_train_smote на X_train_scaled
background = shap.sample(X_train_scaled, 100)
explainer = shap.DeepExplainer(best_model, background)

shap_values = explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):
    sv = shap_values[0]
else:
    sv = shap_values

if len(sv.shape) == 3:
    sv = sv[:, :, 0]

plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_test_scaled, feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
print("График 'shap_summary.png' успешно сохранен в папке проекта!")