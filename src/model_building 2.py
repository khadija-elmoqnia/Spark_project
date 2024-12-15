from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import SparkSession
import pandas as pd
from spark_session import create_spark_session

def train_ridge_model(train_data):
    # Préparer les caractéristiques pour le modèle
    assembler = VectorAssembler(inputCols=["Sensor measurement 11", "Sensor measurement 12", "Sensor measurement 13", 
                                           "Sensor measurement 15", "Sensor measurement 17", "Sensor measurement 2", 
                                           "Sensor measurement 20", "Sensor measurement 21", "Sensor measurement 3", 
                                           "Sensor measurement 4", "Sensor measurement 7", "Sensor measurement 8"], 
                                outputCol="features")
    train_data = assembler.transform(train_data)
    
    # Initialiser et entraîner le modèle de régression Ridge (utilise la régularisation L2)
    ridge = GeneralizedLinearRegression(featuresCol="features", labelCol="TTE", family="gaussian", 
                                        regParam=0.1, link="identity")
    model = ridge.fit(train_data)
    return model

if __name__ == "__main__":
    # Utiliser la session Spark partagée
    spark = create_spark_session()
    
    print("La session Spark fonctionne en mode:", spark.sparkContext.master)
    
    # Charger le fichier Excel avec pandas
    pandas_df = pd.read_excel("C:/Users/Admin/Desktop/spark_project/data/selected_features.xlsx")  # Charger le fichier Excel

    # Convertir le DataFrame pandas en DataFrame Spark
    train_set = spark.createDataFrame(pandas_df)

    # Vérifier le schéma pour s'assurer des noms de colonnes corrects
    train_set.printSchema()

    # Entraîner le modèle Ridge
    ridge_model = train_ridge_model(train_set)

    # Sauvegarder le modèle
    model_output_path = "file:///C:/Users/Admin/Desktop/spark_project/models/ridge_model"
    
    # Sauvegarder le modèle dans le système de fichiers local
    ridge_model.write().overwrite().save(model_output_path)
    
    print("Le modèle Ridge a été sauvegardé avec succès.")
