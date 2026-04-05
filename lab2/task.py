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
import warnings
warnings.filterwarnings('ignore')

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

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

old_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
old_model.fit(X_train, y_train)
y_pred_old = old_model.predict(X_test)
y_pred_proba_old = old_model.predict_proba(X_test)[:, 1]


print("\nСтарая модель (Decision Tree без настроек)")
print("\nКлассификационный отчет:")
print(classification_report(y_test, y_pred_old, target_names=['Лояльные', 'Ушедшие']))

cm_old = confusion_matrix(y_test, y_pred_old)
ConfusionMatrixDisplay(confusion_matrix=cm_old, display_labels=['Лояльные', 'Ушедшие']).plot(cmap='Blues')
plt.title('Ошибки старой модели')
plt.show()

churn_rate = y_train.mean()
print(f"\nДоля ушедших в данных: {churn_rate:.1%}")
print(f"Доля лояльных: {1-churn_rate:.1%}")

tn, fp, fn, tp = cm_old.ravel()
print(f"\nОшибки старой модели:")
print(f"  - Ложные срабатывания (FP): {fp}")
print(f"  - Пропуски (FN): {fn}")

print(f"\nДля бизнеса важнее находить ушедших (FN критичнее чем FP)")
print(f"Цель: поднять Recall для класса 'Ушедшие' до 70%\n")

print("\nЭксперименты с разными подходами")


model_weighted = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
])


model_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

print("Созданы две модели:")
print("  - с автоматической настройкой весов")
print("  - с дополнением данных через SMOTE\n")
print("\nПодбор гиперпараметров")

param_grid = {
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4, 8],
    'classifier__criterion': ['gini', 'entropy']
}

search = GridSearchCV(
    model_weighted,
    param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    verbose=0
)

print("Ищем лучшие параметры...")
search.fit(X_train, y_train)

print(f"\nЛучшие параметры:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Средний Recall на кросс-валидации: {search.best_score_:.3f}\n")



print("\nОценка финальной модели")

best_model = search.best_estimator_
best_model.fit(X_train, y_train)

y_pred_new = best_model.predict(X_test)
y_pred_proba_new = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred_new)
precision = precision_score(y_test, y_pred_new)
recall = recall_score(y_test, y_pred_new)
f1 = f1_score(y_test, y_pred_new)
auc_roc = roc_auc_score(y_test, y_pred_proba_new)

print("\nМетрики новой модели:")
print(f"  Точность (Accuracy):  {accuracy:.3f}")
print(f"  Точность (Precision): {precision:.3f}")
print(f"  Полнота (Recall):     {recall:.3f}  <-- главный показатель")
print(f"  F1-мера:              {f1:.3f}")
print(f"  AUC-ROC:              {auc_roc:.3f}")

print("\nДетальный отчет:")
print(classification_report(y_test, y_pred_new, target_names=['Лояльные', 'Ушедшие']))

cm_new = confusion_matrix(y_test, y_pred_new)
ConfusionMatrixDisplay(confusion_matrix=cm_new, display_labels=['Лояльные', 'Ушедшие']).plot(cmap='Greens')
plt.title('Ошибки новой модели')
plt.show()

plt.figure(figsize=(8, 6))
fpr_old, tpr_old, _ = roc_curve(y_test, y_pred_proba_old)
fpr_new, tpr_new, _ = roc_curve(y_test, y_pred_proba_new)

plt.plot(fpr_old, tpr_old, '--', label=f'Старая (AUC={roc_auc_score(y_test, y_pred_proba_old):.3f})')
plt.plot(fpr_new, tpr_new, label=f'Новая (AUC={auc_roc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайно')
plt.xlabel('Ложные срабатывания (FPR)')
plt.ylabel('Верные попадания (TPR)')
plt.title('ROC-кривые')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("\nЧто влияет на решение модели")


feature_names = (numerical_cols + 
                 list(best_model.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_cols)))

importance = best_model.named_steps['classifier'].feature_importances_
imp_df = pd.DataFrame({'Признак': feature_names, 'Важность': importance})
imp_df = imp_df.sort_values('Важность', ascending=False)

print("\nТоп-10 важных признаков:")
for i, row in imp_df.head(10).iterrows():
    print(f"  {row['Признак']:30} {row['Важность']:.3f}")

plt.figure(figsize=(10, 7))
top10 = imp_df.head(10)
plt.barh(range(len(top10)), top10['Важность'], color='steelblue')
plt.yticks(range(len(top10)), top10['Признак'])
plt.xlabel('Важность')
plt.title('Какие признаки важнее всего')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


print("\nСравнение старой и новой модели")

print(f"\n{'Показатель':<20} {'Старая':>10} {'Новая':>10} {'Изменение':>10}")
print("-"*50)
print(f"{'Accuracy':<20} {accuracy_score(y_test, y_pred_old):>10.3f} {accuracy:>10.3f} {accuracy - accuracy_score(y_test, y_pred_old):>+9.3f}")
print(f"{'Precision':<20} {precision_score(y_test, y_pred_old):>10.3f} {precision:>10.3f} {precision - precision_score(y_test, y_pred_old):>+9.3f}")
print(f"{'Recall':<20} {recall_score(y_test, y_pred_old):>10.3f} {recall:>10.3f} {recall - recall_score(y_test, y_pred_old):>+9.3f}")
print(f"{'F1-score':<20} {f1_score(y_test, y_pred_old):>10.3f} {f1:>10.3f} {f1 - f1_score(y_test, y_pred_old):>+9.3f}")
print(f"{'AUC-ROC':<20} {roc_auc_score(y_test, y_pred_proba_old):>10.3f} {auc_roc:>10.3f} {auc_roc - roc_auc_score(y_test, y_pred_proba_old):>+9.3f}")

# Детали ошибок
tn_old, fp_old, fn_old, tp_old = cm_old.ravel()
tn_new, fp_new, fn_new, tp_new = cm_new.ravel()

print(f"\n{'Тип ошибки':<25} {'Старая':>10} {'Новая':>10} {'Изменение':>10}")
print(f"{'False Positives (FP)':<25} {fp_old:>10} {fp_new:>10} {fp_new - fp_old:>+10}")
print(f"{'False Negatives (FN)':<25} {fn_old:>10} {fn_new:>10} {fn_new - fn_old:>+10}")
print(f"{'Всего ошибок':<25} {fp_old+fn_old:>10} {fp_new+fn_new:>10} {(fp_new+fn_new)-(fp_old+fn_old):>+10}")
print("\nИтоги")

if recall >= 0.70:
    print(f"Цель достигнута. Recall = {recall:.1%} (требовалось >=70%)")
else:
    print(f"Цель не достигнута. Recall = {recall:.1%} (требовалось 70%)")

print(f"\nЧто улучшилось:")
print(f" Полнота (Recall) выросла на {recall - recall_score(y_test, y_pred_old):.1%}")
print(f" Пропущенных уходов стало на {fn_old - fn_new} меньше")
print(f" Компания сможет удержать больше клиентов")

if fp_new > fp_old:
    print(f"\nС другой стороны:")
    print(f" Ложных срабатываний стало на {fp_new - fp_old} больше")
    