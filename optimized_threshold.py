import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from tqdm import tqdm

# 1. Загружаем тест и модель
test_df = pd.read_csv("data/df_test.csv").dropna(subset=["text"])
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
cb_model = CatBoostClassifier().load_model("models/cb.cbm")

X_test_tfidf = tfidf.transform(test_df["text"])
y_true = test_df["label"].to_numpy()

# 2. Получаем матрицы ВЕРОЯТНОСТЕЙ для всего теста
cb_probs = cb_model.predict_proba(X_test_tfidf)

# Находим топ-1 и топ-2 вероятности для каждого предсказания
sorted_probs = np.sort(cb_probs, axis=1)
top1_probs = sorted_probs[:, -1]
top2_probs = sorted_probs[:, -2]

# Считаем уверенность (разницу между 1 и 2 местом)
margins = top1_probs - top2_probs
preds = np.argmax(cb_probs, axis=1)

print("="*70)
print("  🔍 АНАЛИЗ ПОРОГОВ УВЕРЕННОСТИ CATBOOST ДЛЯ КАСКАДА")
print("="*70)
print(f"{'Порог Margin':<15} | {'% забранного трафика':<22} | {'Accuracy на этом трафике':<25}")
print("-"*70)

# Перебираем пороги разницы вероятностей
best_threshold = 0.5
for th in tqdm(np.linspace(0.3, 0.9, 7)):
    # Фильтруем только те запросы, где CatBoost уверен больше, чем порог th
    confident_mask = margins >= th
    
    if confident_mask.sum() == 0:
        continue
        
    # Считаем метрики для этой «уверенной» части
    acc = (preds[confident_mask] == y_true[confident_mask]).mean()
    traffic_share = confident_mask.mean()
    
    print(f"{th:<15.2f} | {traffic_share:<22.2%} | {acc:<25.2%}")
    
    # Авто-подбор: ищем порог, где точность не ниже 95%
    if acc >= 0.95 and traffic_share > 0.3:
        best_threshold = th

print("="*70)
print(f"💡 Рекомендованный порог для Каскада: {best_threshold:.2f}")
print("="*70)