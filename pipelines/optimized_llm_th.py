import numpy as np
import pandas as pd
import json
import joblib
from sklearn.metrics import accuracy_score

# =====================================================================
# ШАГ 1: Загрузка данных и конфигов
# =====================================================================
# Представим, что мы уже прогнали тест через обе модели и сохранили их вероятности.
# Для расчета нам нужны матрицы вероятностей cb_probs и bert_probs на тесте.
# (В реальном коде ты можешь запустить это прямо внутри файла weights_calc.py)

# Загружаем наш единый конфиг весов, который сделали в прошлый раз
with open("models/metrics.json", "r") as f:
    config = json.load(f)

w_cb = config["weights"]["catboost_weight"]
w_bert = config["weights"]["bert_weight"]

# Допустим, мы определили на прошлом шаге, что cb_threshold = 0.65
CB_THRESHOLD = 0.5 

# !!! ВАЖНО: Ниже заглушка, в твоем коде это реальные матрицы cb_probs и bert_probs !!!
# Их размер должен быть (6544, количество_классов)
# Для примера сгенерируем их, но ты подставь свои выходы моделей:
num_samples = 6544
num_classes = 20
y_true = np.random.randint(0, num_classes, num_samples) # Твои реальные test_df["label"]

# Вектора предсказаний (замени на реальные cb_model.predict_proba и т.д.)
cb_probs = np.random.dirichlet(np.ones(num_classes), size=num_samples)
bert_probs = np.random.dirichlet(np.ones(num_classes), size=num_samples)

# =====================================================================
# ШАГ 2: Отсекаем то, что забрал CatBoost на Этапе 1
# =====================================================================
sorted_cb = np.sort(cb_probs, axis=1)
cb_margins = sorted_cb[:, -1] - sorted_cb[:, -2]

# Маска для запросов, которые ушли на ЭТАП 2 (CatBoost НЕ уверен)
stage2_mask = cb_margins < CB_THRESHOLD

logger_cb_answered = ~stage2_mask
cb_preds = np.argmax(cb_probs, axis=1)

# Точность CatBoost на его "уверенном" куске
cb_confident_acc = accuracy_score(y_true[logger_cb_answered], cb_preds[logger_cb_answered])
print(f"Трафик, забранный CatBoost: {logger_cb_answered.mean():.2%}, Точность: {cb_confident_acc:.2%}")

# =====================================================================
# ШАГ 3: Считаем гибрид для остатков трафика
# =====================================================================
# Берем только те строки, которые дошли до BERTа
hybrid_probs = (cb_probs[stage2_mask] * w_cb) + (bert_probs[stage2_mask] * w_bert)
y_true_stage2 = y_true[stage2_mask]
hybrid_preds = np.argmax(hybrid_probs, axis=1)

# Максимальная гибридная вероятность для каждого "сложного" объекта
max_hybrid_probs = np.max(hybrid_probs, axis=1)

# =====================================================================
# ШАГ 4: Перебор порогов для LLM
# =====================================================================
print("\n" + "="*75)
print(f"{'Порог LLM':<12} | {'В LLM (от всего трафика)':<25} | {'Итоговая точность Каскада':<25}")
print("="*75)

# Допущение: LLM отвечает со 100% точностью (или около 98% на проде)
LLM_ACCURACY = 0.98 

best_llm_threshold = 0.50

for th in np.linspace(0.4, 0.9, 11):
    # Если гибридная вероятность ниже th -> отправляем в LLM
    to_llm_mask = max_hybrid_probs < th
    to_ensemble_mask = ~to_llm_mask
    
    # Считаем, сколько это от ОБЩЕГО трафика системы (всех 6544 строк)
    total_samples = len(y_true)
    llm_count = to_llm_mask.sum()
    ensemble_count = to_ensemble_mask.sum()
    cb_count = logger_cb_answered.sum()
    
    # Считаем количество правильных ответов по всей системе:
    # 1. Точные ответы от CatBoost
    right_cb = (cb_preds[logger_cb_answered] == y_true[logger_cb_answered]).sum()
    # 2. Ответы от ансамбля (кто выше порога)
    right_ensemble = (hybrid_preds[to_ensemble_mask] == y_true_stage2[to_ensemble_mask]).sum()
    # 3. Ответы от LLM (кто ниже порога) с учетом её коэф. точности
    right_llm = int(llm_count * LLM_ACCURACY)
    
    # Общая точность каскада
    total_accuracy = (right_cb + right_ensemble + right_llm) / total_samples
    llm_traffic_share = llm_count / total_samples
    
    print(f"{th:<12.2f} | {llm_traffic_share:<25.2%} | {total_accuracy:<25.2%}")
    
    # Бизнес-таргет: Мы хотим общую точность системы >= 96%, минимизируя трафик в LLM
    if total_accuracy >= 0.96 and best_llm_threshold == 0.50:
        best_llm_threshold = th

print("="*75)
print(f"💡 Рекомендованный порог для сброса в LLM: {best_llm_threshold:.2f}")
print("="*75)