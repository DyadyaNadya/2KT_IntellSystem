import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
file = r"C:\Users\Илья\Documents\GitHub\2KT_IntellSystem\lab2\telecom_churn.csv"
cols = [
    "State", "Area code", "International plan", "Voice mail plan",
    "Number vmail messages", "Total day minutes", "Total day calls",
    "Total day charge", "Total eve minutes", "Total eve calls",
    "Total eve charge", "Total night minutes", "Total night calls",
    "Total night charge", "Total intl minutes", "Total intl calls",
    "Total intl charge", "Customer service calls", "Churn"
]

df = pd.read_csv(file, usecols=cols)

print("=" * 80)
print("ДИАГНОСТИКА ПРОБЛЕМЫ")
print("=" * 80)

# Подготовка данных
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

X = df.drop('Churn', axis=1)
y = df['Churn']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Создаем препроцессор
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Загружаем старую модель (Decision Tree без оптимизации)
old_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

old_pipeline.fit(X_train, y_train)
y_pred_old = old_pipeline.predict(X_test)
y_pred_proba_old = old_pipeline.predict_proba(X_test)[:, 1]

# Диагностика старой модели
print("\n1. Оценка старой модели (Decision Tree без оптимизации):")
print("-" * 60)
print(classification_report(y_test, y_pred_old, target_names=['Лояльные (0)', 'Ушедшие (1)']))

# Матрица ошибок
cm_old = confusion_matrix(y_test, y_pred_old)
disp_old = ConfusionMatrixDisplay(confusion_matrix=cm_old, display_labels=['Лояльные', 'Ушедшие'])
disp_old.plot(cmap='Blues')
plt.title('Матрица ошибок старой модели (Decision Tree)')
plt.show()

# Анализ дисбаланса
churn_rate = y_train.mean()
print(f"\n2. Анализ дисбаланса классов:")
print(f"   Доля ушедших в обучающей выборке: {churn_rate:.2%}")
print(f"   Доля лояльных: {1-churn_rate:.2%}")

# Анализ ошибок
tn, fp, fn, tp = cm_old.ravel()
print(f"\n3. Типы ошибок старой модели:")
print(f"   False Positives (FP - ошибочно предсказали уход): {fp}")
print(f"   False Negatives (FN - пропустили уход): {fn}")

# Определяем критичность ошибок для бизнеса
print(f"\n4. Бизнес-анализ ошибок:")
print(f"   Стоимость FP: Компания тратит ресурсы на удержание лояльного клиента")
print(f"   Стоимость FN: Компания теряет клиента, который мог бы остаться")
print(f"   Критичнее: False Negatives (FN), так как потеря клиента приносит больший убыток")
print(f"\n   Цель оптимизации: Повысить Recall для класса 'Ушедшие' до 0.70")

print("\n" + "=" * 80)
print("СОЗДАНИЕ НАДЕЖНОГО КОНВЕЙЕРА")
print("=" * 80)

# Вариант 1: Decision Tree с class_weight='balanced'
print("\n1. Создание конвейера с class_weight='balanced':")
pipeline_balanced = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
])

# Вариант 2: Decision Tree с SMOTE
print("\n2. Создание конвейера с SMOTE:")
pipeline_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

print("\nОба конвейера созданы успешно!")

print("\n" + "=" * 80)
print("СИСТЕМНЫЙ ПОИСК ЛУЧШИХ ПАРАМЕТРОВ")
print("=" * 80)

# Определяем сетку гиперпараметров для Decision Tree
param_grid = {
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4, 8],
    'classifier__criterion': ['gini', 'entropy']
}

# Используем pipeline с class_weight='balanced' для поиска
grid_search = GridSearchCV(
    pipeline_balanced,
    param_grid,
    cv=5,
    scoring='recall',  # используем recall, так как цель - повысить обнаружение ушедших
    n_jobs=-1,
    verbose=1
)

print("\nПоиск лучших гиперпараметров (может занять время)...")
grid_search.fit(X_train, y_train)

print(f"\nРезультаты GridSearchCV:")
print(f"  Лучшие параметры: {grid_search.best_params_}")
print(f"  Средний Recall на кросс-валидации: {grid_search.best_score_:.4f}")

print("\n" + "=" * 80)
print("ФИНАЛЬНАЯ ОЦЕНКА И ИНТЕРПРЕТАЦИЯ")
print("=" * 80)

# Обучаем финальную модель
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred_new = best_model.predict(X_test)
y_pred_proba_new = best_model.predict_proba(X_test)[:, 1]

# Расчет метрик
accuracy = accuracy_score(y_test, y_pred_new)
precision = precision_score(y_test, y_pred_new)
recall = recall_score(y_test, y_pred_new)
f1 = f1_score(y_test, y_pred_new)
auc_roc = roc_auc_score(y_test, y_pred_proba_new)

print("\n1. Метрики новой оптимизированной модели (Decision Tree):")
print("-" * 60)
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-score:  {f1:.4f}")
print(f"   AUC-ROC:   {auc_roc:.4f}")

print("\n   Полный classification_report:")
print(classification_report(y_test, y_pred_new, target_names=['Лояльные (0)', 'Ушедшие (1)']))

# Матрица ошибок новой модели
cm_new = confusion_matrix(y_test, y_pred_new)
disp_new = ConfusionMatrixDisplay(confusion_matrix=cm_new, display_labels=['Лояльные', 'Ушедшие'])
disp_new.plot(cmap='Greens')
plt.title('Матрица ошибок оптимизированной модели (Decision Tree)')
plt.show()

# ROC-кривая
plt.figure(figsize=(8, 6))
fpr_old, tpr_old, _ = roc_curve(y_test, y_pred_proba_old)
fpr_new, tpr_new, _ = roc_curve(y_test, y_pred_proba_new)

plt.plot(fpr_old, tpr_old, label=f'Старая модель (AUC = {roc_auc_score(y_test, y_pred_proba_old):.3f})', linestyle='--')
plt.plot(fpr_new, tpr_new, label=f'Новая модель (AUC = {auc_roc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые сравнения моделей (Decision Tree)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Важность признаков для Decision Tree
print("\n2. Важность признаков (Decision Tree):")
print("-" * 60)

# Получаем названия признаков после one-hot encoding
feature_names = (numerical_cols + 
                 list(best_model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_cols)))

# Получаем важность признаков
feature_importance = best_model.named_steps['classifier'].feature_importances_

# Создаем DataFrame с важностью признаков
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nТоп-15 наиболее важных признаков:")
print(importance_df.head(15).to_string(index=False))

# Визуализация важности признаков
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Важность признака')
plt.title('Важность признаков (Decision Tree)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Сравнение старой и новой модели
print("\n" + "=" * 80)
print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
print("=" * 80)

comparison_data = {
    'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
    'Старая модель': [
        accuracy_score(y_test, y_pred_old),
        precision_score(y_test, y_pred_old),
        recall_score(y_test, y_pred_old),
        f1_score(y_test, y_pred_old),
        roc_auc_score(y_test, y_pred_proba_old)
    ],
    'Новая модель': [
        accuracy, precision, recall, f1, auc_roc
    ],
    'Изменение': [
        accuracy - accuracy_score(y_test, y_pred_old),
        precision - precision_score(y_test, y_pred_old),
        recall - recall_score(y_test, y_pred_old),
        f1 - f1_score(y_test, y_pred_old),
        auc_roc - roc_auc_score(y_test, y_pred_proba_old)
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Старая модель'] = comparison_df['Старая модель'].apply(lambda x: f"{x:.4f}")
comparison_df['Новая модель'] = comparison_df['Новая модель'].apply(lambda x: f"{x:.4f}")
comparison_df['Изменение'] = comparison_df['Изменение'].apply(lambda x: f"{x:+.4f}")

print("\n", comparison_df.to_string(index=False))

# Анализ матриц ошибок
tn_old, fp_old, fn_old, tp_old = cm_old.ravel()
tn_new, fp_new, fn_new, tp_new = cm_new.ravel()

print("\nДетальный анализ ошибок:")
print("-" * 60)
print(f"Показатель              | Старая модель | Новая модель | Изменение")
print("-" * 60)
print(f"False Positives (FP)    | {fp_old:>12} | {fp_new:>12} | {fp_new - fp_old:>+9}")
print(f"False Negatives (FN)    | {fn_old:>12} | {fn_new:>12} | {fn_new - fn_old:>+9}")
print(f"Всего ошибок            | {fp_old+fn_old:>12} | {fp_new+fn_new:>12} | {(fp_new+fn_new)-(fp_old+fn_old):>+9}")

# Вывод цели достижения
print("\n" + "=" * 80)
print("ВЫВОДЫ")
print("=" * 80)

recall_new = recall_score(y_test, y_pred_new)
if recall_new >= 0.70:
    print(f"✓ Цель достигнута! Recall для класса 'Ушедшие' = {recall_new:.2%} (цель: ≥70%)")
else:
    print(f"✗ Цель не достигнута. Recall = {recall_new:.2%} (цель: 70%)")

print(f"\nКлючевые улучшения:")
print(f"  • Recall вырос на {recall_new - recall_score(y_test, y_pred_old):.2%}")
print(f"  • Количество пропущенных уходов (FN) сократилось на {fn_old - fn_new}")
print(f"  • Цена ошибки: компания сможет удержать больше клиентов")

print(f"\nЛучшие гиперпараметры Decision Tree:")
for param, value in grid_search.best_params_.items():
    print(f"  • {param}: {value}")