import pandas as pd
import joblib
import tensorflow as tf

# 1. ЗАГРУЗКА СОХРАНЕННЫХ АРТЕФАКТОВ
print("Загрузка модели и обработчиков...")
model = tf.keras.models.load_model('disease_risk_model.keras')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# 2. НОВЫЙ ПАЦИЕНТ (пример)
# Сюда приходят данные, например, из веб-формы, которую заполнил врач
new_patient_data = {
    # Здесь должны быть перечислены все ваши 40 колонок
    # Для примера напишу несколько:
    'Age': [45],
    'BMI': [28.5],
    'Sleep_Hours': [6],
    'Physical_Activity': ['Low'], # Текстовый признак
    # ... остальные параметры ...
}

# Если вы хотите протестировать скрипт, можете просто загрузить первую строчку из датасета:
# df = pd.read_csv('dataset.csv')
# new_patient_df = df.drop('Disease_Risk', axis=1).iloc[[0]]

new_patient_df = pd.DataFrame(new_patient_data) # Превращаем словарь в датафрейм

# 3. ПРЕДОБРАБОТКА НОВОГО ПАЦИЕНТА
# Переводим текст в числа теми же энкодерами
for column in new_patient_df.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        new_patient_df[column] = label_encoders[column].transform(new_patient_df[column])

# Масштабируем
patient_scaled = scaler.transform(new_patient_df)

# 4. ПРЕДСКАЗАНИЕ
risk_probability = model.predict(patient_scaled)[0][0]

print("\n=== РЕЗУЛЬТАТ АНАЛИЗА ===")
print(f"Вероятность заболевания: {risk_probability * 100:.1f}%")
if risk_probability > 0.5:
    print("Рекомендация: Требуется дополнительное обследование (Высокий риск).")
else:
    print("Рекомендация: Риск в пределах нормы.")