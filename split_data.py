import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')

le = LabelEncoder()
df["label"] = le.fit_transform(df["label"]) 

df_train,df_test = train_test_split(df,test_size=0.2,stratify=df['label'],random_state=42)



PATH = 'data'
df_train.to_csv(f'{PATH}/df_train.csv')
df_test.to_csv(f'{PATH}/df_test.csv')

joblib.dump(le, "models/label_encoder.pkl")