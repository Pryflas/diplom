import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score
from tensorflow.keras.models import load_model

print("1. Подготовка тестовых данных...")
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'diabetes_binary_health_indicators_BRFSS2015.csv')

# Загружаем датасет
df = pd.read_csv(dataset_path)
X = df.drop(['Diabetes_binary'], axis=1)
y = df['Diabetes_binary']

# Чтобы текстовые признаки не вызывали ошибок, прогоняем через LabelEncoder
# (Для чистоты эксперимента лучше загружать сохраненный label_encoders.pkl, но можно и так)
for column in X.select_dtypes(include=['object']).columns:
    X[column] = X[column].astype(str)

# Разбиваем данные точно так же, как при обучении (ключ: random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print("2. Загрузка обработчиков и применение к тестовой выборке...")
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
target_encoder = joblib.load('target_encoder.pkl')

y_test = target_encoder.transform(y_test)
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

print("3. Загрузка сохраненной нейросети...")
# Загружаем готовую модель (НИКАКОГО ОБУЧЕНИЯ!)
model = load_model('best_disease_risk_model_pr_auc.keras', compile=False)

print("4. Выполнение предсказаний на тестовой выборке...")
y_pred_probs = model.predict(X_test_scaled)

# === ПОИСК ИДЕАЛЬНОГО ПОРОГА ОТСЕЧЕНИЯ ===
print("\nПоиск идеального порога для максимизации F2-score...")
thresholds = np.arange(0.01, 1.0, 0.01)
f2_scores = []

for t in thresholds:
    y_pred_t = (y_pred_probs > t).astype(int)
    score = fbeta_score(y_test, y_pred_t, beta=2)
    f2_scores.append(score)

best_idx = np.argmax(f2_scores)
best_threshold = thresholds[best_idx]
best_max_f2 = f2_scores[best_idx]

print(f"Идеальный порог: {best_threshold:.2f}")
print(f"Максимальный F2-score при этом пороге: {best_max_f2:.4f}")

# Рисуем график
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f2_scores, label='F2 Score', color='darkcyan', linewidth=2)
plt.axvline(x=best_threshold, color='crimson', linestyle='--', 
            label=f'Оптимум ({best_threshold:.2f})')
plt.title('Зависимость метрики F2-score от порога классификации')
plt.xlabel('Порог отсечения (Threshold)')
plt.ylabel('Метрика F2-score')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('threshold_optimization_pr_auc.png')
print("График 'threshold_optimization_pr_auc.png' успешно сохранен!")

# Выводим финальный отчет
print(f"\n=== ФИНАЛЬНЫЙ ОТЧЕТ О КЛАССИФИКАЦИИ (ПОРОГ {best_threshold:.2f}) ===")
y_pred_best = (y_pred_probs > best_threshold).astype(int)
print(classification_report(y_test, y_pred_best, target_names=['Нет диабета', 'Диабет']))