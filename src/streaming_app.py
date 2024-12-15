from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql.functions import from_json


# Créer une session Spark
spark = SparkSession.builder \
    .appName("Kafka Streaming Application") \
    .getOrCreate()

# Définir un schéma pour les données entrantes
schema = StructType([
    StructField('Sensor measurement 11', FloatType(), True),
    StructField('Sensor measurement 12', FloatType(), True),
    StructField('Sensor measurement 13', FloatType(), True),
    StructField('Sensor measurement 15', FloatType(), True),
    StructField('Sensor measurement 17', FloatType(), True),
    StructField('Sensor measurement 2', FloatType(), True),
    StructField('Sensor measurement 20', FloatType(), True),
    StructField('Sensor measurement 21', FloatType(), True),
    StructField('Sensor measurement 3', FloatType(), True),
    StructField('Sensor measurement 4', FloatType(), True),
    StructField('Sensor measurement 7', FloatType(), True),
    StructField('Sensor measurement 8', FloatType(), True)
])

# Lire les données du topic Kafka en streaming
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "sensor-data") \
    .load()

# Désérialiser les données en format JSON et appliquer le schéma
df = df_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", schema).alias("data")) \
    .select("data.*")

# Charger le modèle de régression
model = LinearRegressionModel.load("C:/Users/esi/Desktop/spark_project 1/spark_project/models/regression_model")

# Préparer les données pour la prédiction
assembler = VectorAssembler(inputCols=['Sensor measurement 11', 'Sensor measurement 12', 'Sensor measurement 13', 
                                       'Sensor measurement 15', 'Sensor measurement 17', 'Sensor measurement 2',
                                       'Sensor measurement 20', 'Sensor measurement 21', 'Sensor measurement 3', 
                                       'Sensor measurement 4', 'Sensor measurement 7', 'Sensor measurement 8'], 
                            outputCol="features")

df = assembler.transform(df)

# Appliquer le modèle et effectuer la prédiction
predictions = model.transform(df)

# Sélectionner les prédictions de la colonne TTE
predictions = predictions.select("prediction")

# Afficher les résultats en temps réel
query = predictions.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# Attendre la fin de la lecture en streaming
query.awaitTermination()
