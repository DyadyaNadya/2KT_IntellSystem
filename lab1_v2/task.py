import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os

print("Текущая директория:", os.getcwd())
# Загрузка данных
file = r"C:\Users\Илья\Documents\GitHub\2KT_IntellSystem\lab1_v2\telecom_churn.csv"
cols = [
    "State", "Area code", "International plan", "Voice mail plan",
    "Number vmail messages", "Total day minutes", "Total day calls",
    "Total day charge", "Total eve minutes", "Total eve calls",
    "Total eve charge", "Total night minutes", "Total night calls",
    "Total night charge", "Total intl minutes", "Total intl calls",
    "Total intl charge", "Customer service calls", "Churn"
]

df = pd.read_csv(file, usecols=cols)

print("РАЗБОР ДАННЫХ")


print(f"\n1. Размерность датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

print(f"\n2. Типы данных:")
print(df.dtypes)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n3. Категориальные признаки: {categorical_cols}")
print(f"   Числовые признаки: {numerical_cols}")

print(f"\n4. Пропуски в данных:")
print(df.isnull().sum())

churn_counts = df['Churn'].value_counts()
print(f"\n5. Распределение целевой переменной 'Churn':")
print(f"   Не ушли (False): {churn_counts[False]} клиентов")
print(f"   Ушли (True): {churn_counts[True]} клиентов")
print(f"   Доля оттока: {churn_counts[True] / len(df) * 100:.2f}% ")

print("\nПОДГОТОВКА ДАННЫХ К ОБУЧЕНИЮ")

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]} примеров")
print(f"Размер тестовой выборки: {X_test.shape[0]} примеров")


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)


models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Neighbors': KNeighborsClassifier()
}


print("\nОБУЧЕНИЕ И ПРОВЕРКА МОДЕЛЕЙ")
results = {}
predictions = {}

# for name, model in models.items():
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])
    
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     predictions[name] = y_pred
#     accuracy = accuracy_score(y_test, y_pred)
#     results[name] = accuracy
    
#     print(f"\n{name}:")
#     print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
#     # Дополнительная статистика для лучшей интерпретации
#     if name == 'Logistic Regression':
#         print(f"  Classification Report:")
#         print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))


for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    predictions[name] = y_pred
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{'='*50}")
    print(f"{name}:")
    print(f"{'='*50}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))

print("\nАНАЛИЗ РЕЗУЛЬТАТОВ")


results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False)
print("\nТаблица сравнения моделей:")
print(results_df.to_string(index=False))

plt.figure(figsize=(10, 6))
bars = plt.bar(results_df['Model'], results_df['Accuracy'], 
               color=['#2E86AB', '#A23B72', '#F18F01'])
plt.xlabel('Модель')
plt.ylabel('Accuracy')
plt.title('Сравнение точности моделей для прогнозирования оттока клиентов')
plt.ylim(0, 1)
plt.xticks(rotation=15)

for bar, acc in zip(bars, results_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

best_model = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']
print(f"\nЛучшая модель: {best_model} с точностью {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

print(f"\nМАТРИЦА ОШИБОК ДЛЯ ЛУЧШЕЙ МОДЕЛИ ({best_model})")

best_predictions = predictions[best_model]
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Предсказанные значения')
plt.ylabel('Фактические значения')
plt.title(f'Матрица ошибок - {best_model}')
plt.tight_layout()
plt.show()


tn, fp, fn, tp = cm.ravel()

print(f"\nИнтерпретация матрицы ошибок:")
print(f"  True Negatives (TN): {tn} - правильно предсказанные лояльные клиенты")
print(f"  False Positives (FP): {fp} - ошибочно предсказанные как ушедшие")
print(f"  False Negatives (FN): {fn} - ошибочно предсказанные как лояльные")
print(f"  True Positives (TP): {tp} - правильно предсказанные ушедшие клиенты")

print(f"\nДополнительные метрики для лучшей модели:")
print(f"  Precision (ушедшие): {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Precision: N/A")
print(f"  Recall (ушедшие): {tp/(tp+fn):.4f}" if (tp+fn) > 0 else "  Recall: N/A")
print(f"  Specificity (лояльные): {tn/(tn+fp):.4f}" if (tn+fp) > 0 else "  Specificity: N/A")

print(f"\nВыводы:")
print(f"  - Модель ошибочно предсказала как ушедших (FP): {fp} клиентов")
print(f"  - Модель ошибочно предсказала как лояльных (FN): {fn} клиентов")
print(f"  - Всего ошибок: {fp + fn} из {len(y_test)} клиентов в тестовой выборке")


if best_model == 'Logistic Regression':
    print(f"\n" + "=" * 60)
    print("ВАЖНОСТЬ ПРИЗНАКОВ ДЛЯ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
    print("=" * 60)
    

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    

    feature_names = (numerical_cols + 
                     list(pipeline.named_steps['preprocessor']
                          .named_transformers_['cat']
                          .get_feature_names_out(categorical_cols)))
    

    coefficients = pipeline.named_steps['classifier'].coef_[0]
    

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nТоп-10 наиболее важных признаков:")
    print(feature_importance.head(10).to_string(index=False))