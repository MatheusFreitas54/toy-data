import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('toy-data-platform-main/dlt/data/meta_critic.csv')
df.fillna(0, inplace=True)

# Codificar gêneros como números e normaliza valores
df['Genres'] = LabelEncoder().fit_transform(df['Genres'])

scaler = MinMaxScaler()
df[['Score', 'Number of Reviews']] = scaler.fit_transform(df[['Score', 'Number of Reviews']])

model = keras.Sequential([
   layers.Dense(64, activation='relu', input_shape=(df.shape[1] - 1,)),  
   layers.Dense(32, activation='relu'),
   layers.Dense(1)  # Previsão da nota (score)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

X = df.drop(columns=['Score']) 
y = df['Score'] 

model.fit(X, y, epochs=50, batch_size=8, validation_split=0.2)

novo_filme = [[0, 120, 200, 5]]  
previsao = model.predict(novo_filme)
print(f"Nota prevista: {previsao}")
