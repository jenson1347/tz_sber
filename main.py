import torch
import numpy as np
import torch.nn.functional as F
import joblib
import logging
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

logger = logging.getLogger("HybridPredictor")

class HybridPredictor:
    def __init__(self, catboost_path, tfidf_path, bert_dir, le_path, llm_service=None, device="cpu"):
        """
        Инициализация финального пайплайна.
        
        :param catboost_path: Путь к файлу модели CatBoost (.cbm)
        :param tfidf_path: Путь к сохраненному векторизатору TF-IDF (.pkl)
        :param bert_dir: Папка с сохраненной моделью RuBERT и токенизатором
        :param le_path: Путь к сохраненному LabelEncoder (.pkl)
        :param llm_service: Экземпляр класса GigaChatIntentService (опционально)
        :param device: "mps" для Mac, "cuda" для GPU или "cpu"
        """
        self.device = device
        self.llm_service = llm_service


        with open("models/metrics.json", "r") as f:
            config = json.load(f)

        # Достаем веса на лету
        self.cb_w = config["weights"]["catboost_weight"]
        self.bert_w = config["weights"]["bert_weight"]
        
        logger.info("Загрузка локальных ML/DL компонентов...")
        
        # 1. Загружаем классический ML
        self.catboost = CatBoostClassifier()
        self.catboost.load_model(catboost_path)
        self.tfidf = joblib.load(tfidf_path)
        
        # 2. Загружаем label encoder (чтобы возвращать текст вместо цифр)
        self.le = joblib.load(le_path)
        self.all_intents = list(self.le.classes_)
        
        # 3. Загружаем трансформер
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)
        self.bert_model.eval() # Режим инференса (отключаем dropout)
        
        
        
        logger.info("Все локальные модели загружены")

    def predict(self, text, threshold=0.80, expert_intents=None):
        """
        Многоуровневая гибридная классификация текста.
        
        :param text: Сырой текст запроса от клиента
        :param threshold: Порог уверенности, ниже которого запрос улетает в LLM
        :param cb_weight: Вес прогноза CatBoost в блендинге
        :param bert_weight: Вес прогноза BERT в блендинге
        :param expert_intents: Список названий интентов, где CatBoost сильнее BERT (наш инсайт)
        """
        cb_weight=self.cb_w 
        bert_weight=self.bert_w
        
        # Шаг 1: получение вероятностей catboost
        
        text_tfidf = self.tfidf.transform([text])
        cb_probs = self.catboost.predict_proba(text_tfidf)[0]
        
        cb_best_class_idx = np.argmax(cb_probs)
        cb_confidence = cb_probs[cb_best_class_idx]
        cb_intent_text = self.le.inverse_transform([cb_best_class_idx])[0]
        
        # Шаг 2: проверка увероенности классификатора cb + проверка заполнености данных (существуют ли интенты и список с ними)
        if expert_intents and cb_intent_text in expert_intents and cb_confidence > 0.50:
            return {
                "intent": cb_intent_text,
                "confidence": float(cb_confidence),
                "source": "CatBoost (Expert Rule)",
                "local_f1_saved": True
            }
            
        # Шаг 3: получение вероятностей BERT
        
        inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            bert_probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
        # Шаг 4: блендинг
        final_probs = (cb_probs * cb_weight) + (bert_probs * bert_weight)
        ensemble_class_idx = np.argmax(final_probs)
        ensemble_confidence = final_probs[ensemble_class_idx]
        ensemble_intent_text = self.le.inverse_transform([ensemble_class_idx])[0]
        
        # Шаг 5: проверка порога вызова LLM
        if ensemble_confidence < 0.40 and self.llm_service is not None:
            logger.warning(f"Низкая уверенность ансамбля ({ensemble_confidence:.2%}). Подключаем LLM...")
            
            try:
                # Стучимся в наш изолированный сервис GigaChat
                llm_intent = self.llm_service.get_intent(text, self.all_intents)
                
                # Если LLM вернула валидный интент — отдаем его
                if llm_intent in self.all_intents:
                    return {
                        "intent": llm_intent,
                        "confidence": float(ensemble_confidence), # Оставляем скор ансамбля как маркер сомнения
                        "source": "⚡ GigaChat Service (Guardrail)",
                        "local_f1_saved": False
                    }
            except Exception as e:
                logger.error(f"Ошибка LLM-сервиса, откат на локальный ансамбль. Инфо: {e}")
                
        # Если ансамбль уверен ИЛИ LLM упала/выдала бред — возвращаем локальный блендинг
        return {
            "intent": ensemble_intent_text,
            "confidence": float(ensemble_confidence),
            "source": "🤖 Local Ensemble (BERT + CatBoost)",
            "local_f1_saved": False
        }