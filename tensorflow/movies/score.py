import pandas as pd
import tensorflow as tf
from minio import Minio

client = Minio('localhost:9000', access_key='root', secret_key='rootrootroot', secure=False)
client.fget_object('bucket-name', 'toy-data-platform-main/dlt/data/meta_critic.csv', 'meta_critic.csv')

df = pd.read_csv('meta_critic.csv')
df.fillna(0, inplace=True)
df['Genres'] = df['Genres'].astype('category').cat.codes

X = df.drop('avaliacao', axis=1) 
y = df['avaliacao']

modelo_carregado = tf.keras.models.load_model('meu_modelo.h5')
novos_dados = X.iloc[:5] 
novas_previsoes = modelo_carregado.predict(novos_dados)

print("Notas previstas (escala original):")
print(novas_previsoes * 10) 