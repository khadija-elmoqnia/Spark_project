# spark_session.py
from pyspark.sql import SparkSession

# Fonction pour créer une SparkSession
def create_spark_session(log_level="WARN"):
    # Créer ou récupérer une session Spark
    spark = SparkSession.builder \
        .appName("Creation_Model").master("local[*]") \
        .getOrCreate()
    
    # Réglez le niveau de journalisation dynamiquement
    spark.sparkContext.setLogLevel(log_level)  # Log niveau peut être passé en paramètre
    
    return spark


