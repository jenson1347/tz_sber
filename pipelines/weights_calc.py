import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Выбор девайса для BERTа (Мак на MPS или CPU)

# =====================================================================
# ШАГ 1: Загрузка тестового датасета и компонентов
# =====================================================================

# Укажи тут точное имя своего файла (test.csv или test_df.csv)
test_path = "data/df_test.csv"
test_df = pd.read_csv(test_path)

# Очистка от возможных NaN, которую мы победили в прошлый раз
test_df = test_df.dropna(subset=["text"])
test_texts = test_df["text"].astype(str).tolist()
test_labels = test_df["label"].tolist()



# Загружаем LabelEncoder
le = joblib.load("models/label_encoder.pkl")
class_names = list(le.classes_)

# =====================================================================
# ШАГ 2: Оценка модели CatBoost
# =====================================================================

tfidf = joblib.load("models/tfidf_vectorizer.pkl")
cb_model = CatBoostClassifier().load_model("models/cb.cbm")

# Векторизуем текст и делаем предсказание
X_test_tfidf = tfidf.transform(test_texts)
cb_preds = cb_model.predict(X_test_tfidf).flatten()

# Фиксируем точность
cb_accuracy = accuracy_score(test_labels, cb_preds)


# =====================================================================
# ШАГ 3: Оценка модели RuBERT (Исправленная версия)
# =====================================================================
bert_dir = "models/fine_tuned_rubert_tiny2"
tokenizer = AutoTokenizer.from_pretrained(bert_dir)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
bert_model.eval()

# ИСПРАВЛЕНИЕ: Добавляем return_tensors="pt", чтобы сразу получить тензоры PyTorch
inputs = tokenizer(
    test_texts, 
    padding=True, 
    truncation=True, 
    max_length=64, 
    return_tensors="pt"
)

bert_preds = []
batch_size = 16
num_samples = len(test_texts)

with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        # ИСПРАВЛЕНИЕ: Правильно нарезаем тензоры по первой размерности (батч-директива)
        batch_inputs = {
            k: v[i : i + batch_size]
            for k, v in inputs.items()
        }
        
        outputs = bert_model(**batch_inputs)
        
        # Забираем самый уверенный класс
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        bert_preds.extend(preds)

# Теперь длины гарантированно совпадут!

bert_accuracy = accuracy_score(test_labels, bert_preds)

# =====================================================================
# ШАГ 4: Расчет весов через Softmax
# =====================================================================

def get_softmax_weights(acc_cb, acc_bert, temperature=1.0):
    # Умножаем на 10 для четкой чувствительности к сотым долям
    logits = np.array([acc_cb, acc_bert]) * 10 / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

# Получаем наши заветные веса
cb_weight, bert_weight = get_softmax_weights(cb_accuracy, bert_accuracy, temperature=1.0)

# Сохраняем метрики в JSON, чтобы main.py мог их прочитать
ensemble_config = {
    "metrics": {
        "catboost_accuracy": float(cb_accuracy),
        "bert_accuracy": float(bert_accuracy)
    },
    "weights": {
        "catboost_weight": float(cb_weight),
        "bert_weight": float(bert_weight)
    },
    "total_test_samples": int(len(test_labels)) # добавим для солидности объем теста
}

config_path = "models/metrics.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(ensemble_config, f, indent=4, ensure_ascii=False)