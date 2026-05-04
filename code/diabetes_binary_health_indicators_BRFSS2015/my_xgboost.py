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
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

import xgboost as xgb

# === 3. ОБУЧЕНИЕ XGBOOST ===
print("\nСоздание и обучение градиентного бустинга (XGBoost)...")

# Вычисляем баланс классов для XGBoost (соотношение здоровых к больным)
neg_cases = np.sum(y_train == 0)
pos_cases = np.sum(y_train == 1)
scale_ratio = neg_cases / pos_cases

# Настройка гиперпараметров модели
xgb_model = xgb.XGBClassifier(
    n_estimators=1500,          # Максимальное количество деревьев
    learning_rate=0.02,         # Медленный и аккуратный шаг обучения
    max_depth=6,                # Глубина каждого дерева (позволяет находить сложные связи)
    subsample=0.8,              # Использование 80% данных для каждого дерева (защита от переобучения)
    colsample_bytree=0.8,       # Использование 80% колонок для каждого дерева
    scale_pos_weight=scale_ratio, # Встроенная балансировка классов
    eval_metric='auc',          # Прямая оптимизация ROC-AUC
    early_stopping_rounds=50,   # Остановка, если AUC на валидации не растет 50 шагов
    random_state=42,
    n_jobs=-1                   # Использовать все ядра процессора
)

# Обучение
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
    verbose=50
)

# Сохраняем модель (вместо .keras используется родной формат xgboost или joblib)
joblib.dump(xgb_model, 'xgboost_disease_model.pkl')
print("\nМодель XGBoost успешно обучена и сохранена!")


# === 4. ОЦЕНКА МЕТРИК ===
print("\n=== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
# Получаем вероятности (берем колонку 1, т.к. predict_proba возвращает вероятности для обоих классов)
y_pred_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Оценка AUC-ROC
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"ROC-AUC: {auc_score:.4f}")

y_pred_classes = (y_pred_probs > 0.5).astype(int)
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred_classes, target_names=['Нет диабета', 'Диабет']))

# === 5. ВАЖНОСТЬ ПРИЗНАКОВ ===
print("\nПостроение графика важности признаков...")
plt.figure(figsize=(10, 8))
# Встроенный в XGBoost удобный плоттер важности
xgb.plot_importance(xgb_model, max_num_features=15, importance_type='weight')
plt.title('Топ-15 самых важных признаков для предсказания')
plt.tight_layout()
plt.savefig('xgboost_importance.png')
print("График 'xgboost_importance.png' сохранен!")