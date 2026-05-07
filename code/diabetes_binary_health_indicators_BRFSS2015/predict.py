import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# === 1. НАСТРОЙКИ СИСТЕМЫ ===
THRESHOLD = 0.46  # Тот самый идеальный порог для PR-AUC, который мы нашли
MODEL_NAME = 'best_disease_risk_model_pr_auc.keras'

print("="*60)
print(" 🩺 СИСТЕМА ПРЕДИКТИВНОГО АНАЛИЗА РИСКА ДИАБЕТА (BRFSS 2015) ")
print("="*60)
print("Инициализация системы (загрузка нейросети)...")

# === 2. ЗАГРУЗКА АРТЕФАКТОВ ===
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'diabetes_binary_health_indicators_BRFSS2015.csv')

try:
    model = load_model(MODEL_NAME, compile=False)
    imputer = joblib.load('imputer.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"\n[ОШИБКА] Не удалось загрузить модель или файлы препроцессинга: {e}")
    exit()

try:
    df_headers = pd.read_csv(dataset_path, nrows=0)
    feature_columns = df_headers.drop(['Diabetes_binary'], axis=1).columns.tolist()
except FileNotFoundError:
    print("\n[ОШИБКА] Файл датасета не найден. Он нужен для получения списка колонок.")
    exit()

patient_data = {}

# === 3. ПОЛНОЕ АНКЕТИРОВАНИЕ ПАЦИЕНТА (21 ВОПРОС) ===
print("\nПожалуйста, заполните анкету пациента (вводите числа):")
print("-" * 60)

try:
    # 1. Базовые параметры и здоровье
    print("\n--- БАЗОВЫЕ ПОКАЗАТЕЛИ ---")
    patient_data['Age'] = float(input("Возрастная категория (1: 18-24, 7: 45-49, 10: 60-64, 13: 80+): "))
    patient_data['Sex'] = float(input("Пол (0: Женщина, 1: Мужчина): "))
    patient_data['BMI'] = float(input("Индекс массы тела (ИМТ, например 25.5): "))
    patient_data['GenHlth'] = float(input("Общее здоровье (1: Отлично ... 5: Плохо): "))
    
    # 2. Хронические заболевания
    print("\n--- ИСТОРИЯ БОЛЕЗНЕЙ ---")
    patient_data['HighBP'] = float(input("Повышенное давление? (0: Нет, 1: Да): "))
    patient_data['HighChol'] = float(input("Повышенный холестерин? (0: Нет, 1: Да): "))
    patient_data['CholCheck'] = float(input("Сдавали холестерин за последние 5 лет? (0: Нет, 1: Да): "))
    patient_data['Stroke'] = float(input("Был ли инсульт? (0: Нет, 1: Да): "))
    patient_data['HeartDiseaseorAttack'] = float(input("Ишемическая болезнь сердца или инфаркт? (0: Нет, 1: Да): "))
    patient_data['DiffWalk'] = float(input("Трудности при ходьбе/подъеме по лестнице? (0: Нет, 1: Да): "))

    # 3. Образ жизни
    print("\n--- ОБРАЗ ЖИЗНИ ---")
    patient_data['Smoker'] = float(input("Выкурили больше 100 сигарет за жизнь? (0: Нет, 1: Да): "))
    patient_data['HvyAlcoholConsump'] = float(input("Много ли пьете алкоголя? (>14 дринков в неделю) (0: Нет, 1: Да): "))
    patient_data['PhysActivity'] = float(input("Была ли физ. активность за последние 30 дней? (0: Нет, 1: Да): "))
    patient_data['Fruits'] = float(input("Употребляете фрукты каждый день? (0: Нет, 1: Да): "))
    patient_data['Veggies'] = float(input("Употребляете овощи каждый день? (0: Нет, 1: Да): "))

    # 4. Ментальное и физическое состояние (дни)
    print("\n--- САМОЧУВСТВИЕ ЗА ПОСЛЕДНИЕ 30 ДНЕЙ ---")
    patient_data['MentHlth'] = float(input("Сколько дней было плохое ментальное здоровье? (от 0 до 30): "))
    patient_data['PhysHlth'] = float(input("Сколько дней было плохое физическое здоровье (травмы/болезни)? (от 0 до 30): "))

    # 5. Социальные факторы
    print("\n--- СОЦИАЛЬНЫЕ ФАКТОРЫ ---")
    patient_data['AnyHealthcare'] = float(input("Есть ли мед. страховка? (0: Нет, 1: Да): "))
    patient_data['NoDocbcCost'] = float(input("Отказывались ли от врача из-за нехватки денег за год? (0: Нет, 1: Да): "))
    patient_data['Education'] = float(input("Уровень образования (1: Без школы ... 6: Высшее): "))
    patient_data['Income'] = float(input("Уровень дохода (1: Меньше $10k ... 8: Больше $75k): "))

except ValueError:
    print("\n[ОШИБКА] Нужно вводить только цифры. Перезапустите скрипт.")
    exit()

# === 4. ОБРАБОТКА И ПРЕДСКАЗАНИЕ ===
print("\n" + "="*60)
print(" Анализ данных...")

# Важно: располагаем колонки точно в том порядке, в каком они были при обучении
patient_df = pd.DataFrame([patient_data], columns=feature_columns)

patient_imputed = imputer.transform(patient_df)
patient_scaled = scaler.transform(patient_imputed)

risk_probability = model.predict(patient_scaled, verbose=0)[0][0]

# === 5. ВЫВОД РЕЗУЛЬТАТА ===
print("="*60)
print(" РЕЗУЛЬТАТЫ АНАЛИЗА НЕЙРОСЕТЬЮ")
print("="*60)

print(f"Вероятность наличия диабета: {risk_probability * 100:.1f}%")
print(f"Клинический порог отсечения: {THRESHOLD * 100:.1f}%\n")

if risk_probability > THRESHOLD:
    print("⚠️ ВЕРДИКТ: ПОВЫШЕННЫЙ РИСК (Класс 1: Диабет)")
    print("РЕКОМЕНДАЦИЯ: Алгоритм выявил высокую вероятность заболевания.")
    print("Необходимо направить пациента на клинические анализы (HbA1c, глюкоза натощак).")
else:
    print("✅ ВЕРДИКТ: РИСК В ПРЕДЕЛАХ НОРМЫ (Класс 0: Нет диабета)")
    print("РЕКОМЕНДАЦИЯ: Показатели стабильны. Скрининг пройден успешно.")
print("="*60)