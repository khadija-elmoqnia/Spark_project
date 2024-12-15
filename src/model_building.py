from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
import pandas as pd
from spark_session import create_spark_session

def train_model(train_data):
    # Préparer les caractéristiques pour le modèle
    assembler = VectorAssembler(inputCols=["Sensor measurement 11", "Sensor measurement 12", "Sensor measurement 13", 
                                           "Sensor measurement 15", "Sensor measurement 17", "Sensor measurement 2", 
                                           "Sensor measurement 20", "Sensor measurement 21", "Sensor measurement 3", 
                                           "Sensor measurement 4", "Sensor measurement 7", "Sensor measurement 8"], 
                                outputCol="features")
    train_data = assembler.transform(train_data)
    
    # Initialiser le modèle de régression linéaire
    lr = LinearRegression(featuresCol="features", labelCol="TTE")
    
    # Définir les paramètres à tester dans la recherche de grille
    param_grid = (ParamGridBuilder()
                  .addGrid(lr.regParam, [0.01, 0.1, 0.5])  # Régularisation L2 (paramètre de pénalisation)
                  .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])  # ElasticNet (combiné L1 et L2)
                  .addGrid(lr.maxIter, [10, 50, 100])  # Nombre d'itérations
                  .build())
    
    # Définir l'évaluateur pour la régression
    evaluator = RegressionEvaluator(labelCol="TTE", predictionCol="prediction", metricName="rmse")
    
    # Configurer la validation croisée
    crossval = CrossValidator(estimator=lr,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=5)  # Validation croisée à 5 plis
    
    # Appliquer la validation croisée et entraîner le modèle
    model = crossval.fit(train_data)
    
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

    # Entraîner le modèle avec recherche d'hyperparamètres
    model = train_model(train_set)

    # Sauvegarder le modèle
    model_output_path = "file:///C:/Users/Admin/Desktop/spark_project/models/regression_model"
    
    # Sauvegarder le meilleur modèle trouvé
    model.bestModel.write().overwrite().save(model_output_path)
    
    print("Le modèle a été sauvegardé avec succès.")
