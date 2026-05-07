import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, fbeta_score
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
import shap

# === ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ ===
def f2_score(y_true, y_pred):
    """Метрика F2-score (Полнота важнее точности)"""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(K.round(y_pred), 'float32') 
    
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    
    f2 = (5 * precision * recall) / (4 * precision + recall + K.epsilon())
    return f2

class CustomScoreCallback(tf.keras.callbacks.Callback):
    """Вычисляет взвешенную метрику (40% AUC + 60% Recall) просто для информации"""
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            val_auc = logs.get('val_AUC', 0)
            val_recall = logs.get('val_recall', 0)
            custom_score = (0.4 * val_auc) + (0.6 * val_recall)
            logs['val_custom_score'] = custom_score
            # print(f" — val_custom_score: {custom_score:.4f}")

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
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    
    for i in range(hp.Int('num_layers', 1, 7)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
        
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(BatchNormalization())
            
        model.add(Dropout(
            rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        ))
    
    model.add(Dense(1, activation='sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    # Сеть считает все метрики
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='AUC'), 
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.Recall(name='recall'),
            f2_score
        ]
    )
    
    return model

# ИЗМЕНЕНИЕ: ТЮНЕР НАСТРОЕН НА F2-SCORE (val_f2_score)
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_f2_score", direction="max"), 
    max_epochs=50,
    factor=3,
    directory='tuner_logs_f2',            # Изменено название папки
    project_name='diabetes_tuning_f2',    # Изменено название проекта
    overwrite=True
)

# ИЗМЕНЕНИЕ: КОЛЛБЭК РАННЕЙ ОСТАНОВКИ НАСТРОЕН НА F2-SCORE
early_stop = EarlyStopping(monitor='val_f2_score', mode='max', patience=10, restore_best_weights=True)
my_custom_scorer = CustomScoreCallback()

print("Начало поиска лучших гиперпараметров...")
tuner.search(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=512,
    class_weight=class_weights_dict,
    callbacks=[early_stop, my_custom_scorer],
    verbose=1
)

# === 4. ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ ===
print("\n=== ВЫБРАННЫЕ ГИПЕРПАРАМЕТРЫ ===")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

best_model = tuner.hypermodel.build(best_hps)

print("\n=== АРХИТЕКТУРА ЛУЧШЕЙ НЕЙРОСЕТИ ===")
best_model.summary()

# ИЗМЕНЕНИЕ: КОЛЛБЭК УМЕНЬШЕНИЯ ШАГА НАСТРОЕН НА F2-SCORE
reduce_lr = ReduceLROnPlateau(monitor='val_f2_score', mode='max', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

print("\nФинальное обучение лучшей модели...")
history = best_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100, 
    batch_size=512, 
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr, my_custom_scorer], 
    verbose=1
)

# Изменено имя сохраняемой модели
best_model.save('best_disease_risk_model_f2.keras')
print("\nЛучшая модель успешно обучена и сохранена!")

# === 5. ОЦЕНКА МЕТРИК НА ТЕСТОВОЙ ВЫБОРКЕ ===
print("\n=== ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ ===")
y_pred_probs = best_model.predict(X_test_scaled)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# 1. ROC-AUC
auc_score = roc_auc_score(y_test, y_pred_probs)
print(f"ROC-AUC: {auc_score:.4f}")

# 2. PR-AUC
precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_pred_probs)
pr_auc = auc(recall_arr, precision_arr)
print(f"PR-AUC: {pr_auc:.4f}")

# 3. F2-score
f2 = fbeta_score(y_test, y_pred_classes, beta=2)
print(f"F2-Score: {f2:.4f}")

print("\nОтчет о классификации (Порог 0.5):")
print(classification_report(y_test, y_pred_classes, target_names=['Нет диабета', 'Диабет']))

# === 6. ИНТЕРПРЕТАЦИЯ И ГРАФИКИ (SHAP) ===
print("\nПостроение графиков SHAP (это может занять пару минут)...")
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
# Изменено имя сохраняемого графика
plt.savefig('shap_summary_f2.png')
print("График 'shap_summary_f2.png' успешно сохранен в папке проекта!")