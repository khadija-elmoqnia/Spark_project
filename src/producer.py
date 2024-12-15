from kafka import KafkaProducer
import pandas as pd
import time
import json

# Initialisation du producteur Kafka
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Charger le dataset CSV
data = pd.read_csv(r'C:\Users\esi\Desktop\spark_project 1\spark_project\data\train_set.csv')

# Sélectionner les colonnes nécessaires
columns = ['Sensor measurement 11', 'Sensor measurement 12', 'Sensor measurement 13', 
           'Sensor measurement 15', 'Sensor measurement 17', 'Sensor measurement 2',
           'Sensor measurement 20', 'Sensor measurement 21', 'Sensor measurement 3', 
           'Sensor measurement 4', 'Sensor measurement 7', 'Sensor measurement 8']

# Créer un producteur Kafka
def send_data_to_kafka():
    for _, row in data[columns].iterrows():
        # Convertir chaque ligne en dictionnaire
        record = row.to_dict()
        # Envoyer la donnée à Kafka
        producer.send('sensor-data', value=record)
        print(f"Data sent: {record}")
        time.sleep(1)  # Simuler un délai pour les données en temps réel

# Lancer le producteur
send_data_to_kafka()
