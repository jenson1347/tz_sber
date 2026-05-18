from main import HybridPredictor
from services.llm import GigaChatIntentService as llm_backup_service


MY_EXPERT_CLASSES = ["заказ_справки_нфдл", "какой_то_еще_класс_1", "какой_то_еще_класс_2"]


predictor = HybridPredictor(
    catboost_path="models/cb.cbm",
    tfidf_path="models/tfidf_vectorizer.pkl",
    bert_dir="models/fine_tuned_rubert_tiny2",
    le_path="models/label_encoder.pkl",
    llm_service=llm_backup_service(), 
    #device="mps"
)

result = predictor.predict(
    text="уберите 4G из моего телефона", 
    threshold=0.2,
    expert_intents=MY_EXPERT_CLASSES
)

print(f"Определенный интент: {result['intent']}")
print(f"Уверенность (Confidence): {result['confidence']:.2%}")
print(f"Кто принял решение: {result['source']}")