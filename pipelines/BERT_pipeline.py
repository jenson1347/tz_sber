import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from torch import nn
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight



#Загрузка расплитанных дата фреймов для обучения и теста

PATH='data'
df_train = pd.read_csv(f'{PATH}/df_train.csv')
df_test = pd.read_csv(f'{PATH}/df_test.csv')

df_train = df_train.dropna(subset=["text"])
df_test = df_test.dropna(subset=["text"])


num_labels = len(df_test.label.unique())
train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
test_dataset = Dataset.from_pandas(df_test[['text', 'label']])


#Инициализация модели

MODEL_NAME = "cointegrated/rubert-tiny2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


#Так как в выборке присутствие дисабланс интентов, необходимо для каждого из них расчитать вес
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df_train['label']),
    y=df_train['label']
)
class_weights = class_weights.astype(np.float32)
class_weights = torch.from_numpy(class_weights)

# Создаем кастомный Trainer, который учитывает эти веса (некторые методы не работали на маке, пришлось писать свой трейнер,
# наследуется от класса transforems.trainer)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Считаем базовый лосс без весов
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        raw_loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        
        # Переносим метки текущего батча на CPU, чтобы безопасно достать нужные веса
        labels_cpu = labels.view(-1).to("cpu")
        
        # Достаем веса классов на CPU
        batch_weights_cpu = class_weights[labels_cpu]
        
        # Перекидываем готовые веса батча на то устройство, где крутится модель (MPS) - только для MAC нужно
        batch_weights = batch_weights_cpu.to(raw_loss.device)
        # -------------------------------------
        
        # Перемножаем и берем среднее
        weighted_loss = (raw_loss * batch_weights).mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss
    

#Обучение классификационной головы берта

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./results",          
    learning_rate=2e-5,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,
    num_train_epochs=3,              
    weight_decay=0.01,
    eval_strategy="epoch",     
    save_strategy="epoch",           
    load_best_model_at_end=True,     
    logging_steps=10,
    report_to="none"                 
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()


predictions = trainer.predict(tokenized_test)
preds_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

#сохранение модели


model.save_pretrained("models/fine_tuned_rubert_tiny2")
tokenizer.save_pretrained("models/fine_tuned_rubert_tiny2")

