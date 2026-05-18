import pandas as pd
import numpy as np
import re
import joblib
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import catboost as cb
import logging

import optuna
from sklearn.metrics import f1_score


logger = logging.getLogger("GBClassifier pipeline")
logging.basicConfig(level=logging.INFO)
#Загрузка расплитанных дата фреймов для обучения и теста

PATH='data'
df_train = pd.read_csv(f'{PATH}/df_train.csv')
df_test = pd.read_csv(f'{PATH}/df_test.csv')


logger.info("Загружены датасеты")

#Предобработка для tf-idf

morph = pymorphy3.MorphAnalyzer()

def clean_and_lemmatize(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^а-яё ]', ' ', text) 
    
    words = text.split()
    
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    
    return " ".join(lemmatized_words)

#Добавление в фреймы обработанный текст

df_train['cleaned_Text'] = df_train['text'].apply(clean_and_lemmatize)
df_test['cleaned_Text'] = df_test['text'].apply(clean_and_lemmatize)


logger.info("Предобработан текст")

#Разбиение таргета и предобработанного текста на подвыборки (для удобства)

X_train_text = df_train['cleaned_Text']
y_train = df_train['label']

X_test_text = df_test['cleaned_Text']
y_test = df_test['label']


#TF-IDF

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train = tfidf.fit_transform(X_train_text).toarray()
X_test = tfidf.transform(X_test_text).toarray()

logger.info("Применение TF-IDF")

def objective(trial):
    
    params = {
        'iterations': trial.suggest_int('iterations', 100, 600, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'loss_function': 'MultiClass',
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU' 
    }
    
    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=30,
        verbose=False
    )
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    
    return score


study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=5, timeout=600) # таймаут 10 минут максимум

logger.info(f"Best score after learning: {study.best_value}")

best_params = study.best_params
best_params['loss_function'] = 'MultiClass'
best_params['random_seed'] = 42
best_params['verbose'] = 100

model = cb.CatBoostClassifier(**best_params)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

final_preds = model.predict(X_test)
#print(classification_report(y_test, final_preds)) 

model.save_model('models/cb.cbm')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')