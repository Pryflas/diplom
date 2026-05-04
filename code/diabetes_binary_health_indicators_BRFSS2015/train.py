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
import shap

# === 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
print("Загрузка данных...")
current_dir = os.path.dirname(os.path.abspath(__file__))
# Имя нового файла датасета
dataset_path = os.path.join(current_dir, 'diabetes_binary_health_indicators_BRFSS2015.csv')

df = pd.read_csv(dataset_path)

# Новая целевая переменная
target_column = 'Diabetes_binary' 

# В новом датасете нет Patient_ID, удаляем только целевую переменную
X = df.drop([target_column], axis=1)
y = df[target_column]

# Оставляем кодировщик для совместимости (хотя классы уже 0.0 и 1.0)
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

from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

# УБРАЛИ SMOTE! Вместо этого считаем веса классов
print("Вычисление весов классов...")
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(weights))
print(f"Веса классов: {class_weights_dict}")

# === 3. АРХИТЕКТУРА НЕЙРОСЕТИ ===
print("\nСоздание и обучение нейронной сети...")
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    
    # Слегка расширим сеть, так как убрали SMOTE и ей нужно уловить более сложные связи
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4), # Чуть больше Dropout для борьбы с переобучением
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC']) # Метрика AUC напрямую!

# Коллбэки: Ранняя остановка + уменьшение шага обучения
early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

history = model.fit(X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=100, 
                    batch_size=512, # Увеличенный батч для более точного градиента
                    class_weight=class_weights_dict, # Передаем веса
                    callbacks=[early_stop, reduce_lr], 
                    verbose=1)

model.save('disease_risk_model.keras')
print("\nМодель успешно обучена и сохранена!")

# === 4. ОЦЕНКА МЕТРИК ===
print("\n=== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
y_pred_probs = model.predict(X_test_scaled)

# Оценка AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"ROC-AUC: {auc_score:.4f}")

# Подбор оптимального порога
# Т.к. порог 0.5 может не подходить, найдем порог с лучшим балансом (f1-score)
y_pred_classes = (y_pred_probs > 0.5).astype(int)
print("\nОтчет о классификации (Порог 0.5):")
print(classification_report(y_test, y_pred_classes, target_names=['Нет диабета', 'Диабет']))

# # === 2. РАЗБИЕНИЕ И ПРЕДОБРАБОТКА ===
# print("Разбиение и нормализация данных...")
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# imputer = SimpleImputer(strategy='mean')
# X_train_imputed = imputer.fit_transform(X_train)
# X_val_imputed = imputer.transform(X_val)
# X_test_imputed = imputer.transform(X_test)
# joblib.dump(imputer, 'imputer.pkl') 

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_imputed)
# X_val_scaled = scaler.transform(X_val_imputed)
# X_test_scaled = scaler.transform(X_test_imputed)
# joblib.dump(scaler, 'scaler.pkl') 

# print("Применение SMOTE...")
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# # === 3. АРХИТЕКТУРА НЕЙРОСЕТИ ===
# print("\nСоздание и обучение нейронной сети...")
# model = Sequential([
#     Input(shape=(X_train_smote.shape[1],)),
    
#     # Dense(256, activation='relu'),
#     # BatchNormalization(),
#     # Dropout(0.3),
    
#     # Dense(128, activation='relu'),
#     # BatchNormalization(),
#     # Dropout(0.3),
    
#     Dense(64, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
    
#     Dense(32, activation='relu'),
#     Dropout(0.3),
    
#     # ИЗМЕНЕНИЕ: Для бинарной классификации нужен 1 нейрон и функция активации sigmoid
#     Dense(1, activation='sigmoid')
# ])

# # ИЗМЕНЕНИЕ: Функция потерь заменена на binary_crossentropy
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# history = model.fit(X_train_smote, y_train_smote,
#                     validation_data=(X_val_scaled, y_val),
#                     epochs=100, 
#                     batch_size=64, 
#                     callbacks=[early_stop], 
#                     verbose=1)

# model.save('disease_risk_model.keras')
# print("\nМодель успешно обучена и сохранена!")

# # === 4. ОЦЕНКА МЕТРИК ===
# print("\n=== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
# y_pred_probs = model.predict(X_test_scaled)
# # ИЗМЕНЕНИЕ: argmax не используется для sigmoid. Если вероятность > 0.5, то класс 1, иначе 0.
# y_pred_classes = (y_pred_probs > 0.75).astype(int)

# # ИЗМЕНЕНИЕ: Для бинарного roc_auc_score параметр multi_class больше не нужен
# print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_probs):.4f}")
# print("\nОтчет о классификации:")
# print(classification_report(y_test, y_pred_classes, target_names=['Нет диабета', 'Диабет']))

# === 5. ИНТЕРПРЕТАЦИЯ И ГРАФИКИ (SHAP) ===
print("\nПостроение графиков SHAP (это может занять пару минут)...")
background = shap.sample(X_train_smote, 100)
explainer = shap.DeepExplainer(model, background)

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