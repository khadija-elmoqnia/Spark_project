import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator  # Importation de l'évaluateur de régression
from spark_session import create_spark_session 

# Fonction pour prédire à l'aide du modèle
def predict_model(model, test_data):
    # Préparer les caractéristiques pour les données de test
    assembler = VectorAssembler(inputCols=["Sensor measurement 11", "Sensor measurement 12", "Sensor measurement 13", 
                                           "Sensor measurement 15", "Sensor measurement 17", "Sensor measurement 2", 
                                           "Sensor measurement 20", "Sensor measurement 21", "Sensor measurement 3", 
                                           "Sensor measurement 4", "Sensor measurement 7", "Sensor measurement 8"], 
                                outputCol="features")
    test_data = assembler.transform(test_data)
    
    # Faire des prédictions sur les données de test
    predictions = model.transform(test_data)
    return predictions

if __name__ == "__main__":
    # Créer une session Spark
    spark = create_spark_session()  # Cette ligne crée la session Spark
    
    print("La session Spark fonctionne en mode:", spark.sparkContext.master)
    
    # Charger les données de test préparées depuis un fichier Excel
    pandas_df = pd.read_excel("C:/Users/Admin/Desktop/spark_project/data/prepared_test_set.xlsx")  # Fichier de test préparé
    #pandas_df = pd.read_excel("C:/Users/Admin/Desktop/spark_project/data/selected_features.xlsx")

    # Convertir le DataFrame pandas en DataFrame Spark
    test_data = spark.createDataFrame(pandas_df)

    # Vérifier le schéma pour s'assurer des noms de colonnes corrects
    test_data.printSchema()

    # Charger le modèle sauvegardé
    model_path = "file:///C:/Users/Admin/Desktop/spark_project/models/regression_model"
    model = LinearRegressionModel.load(model_path)

    # Prédire à l'aide du modèle sauvegardé
    predictions = predict_model(model, test_data)

    # Afficher les valeurs réelles et prédites
    results = predictions.select("TTE", "prediction")  # TTE est la valeur réelle, prediction est la valeur prédite
    results.show()

    # Collecter les résultats sous forme de DataFrame pandas pour une analyse ou comparaison ultérieure
    results_df = results.toPandas()
    
    # Calculer la différence entre les valeurs réelles et prédites
    results_df['difference'] = results_df['TTE'] - results_df['prediction']
    
    # Afficher les valeurs réelles, prédites et la différence
    print("\nValeurs réelles vs Prédictions avec la différence :")
    print(results_df)

    # Calcul des métriques d'évaluation supplémentaires
    mse = ((results_df['difference']) ** 2).mean()  # Erreur quadratique moyenne (MSE)
    rmse = mse ** 0.5  # Racine de l'erreur quadratique moyenne (RMSE)
    mae = results_df['difference'].abs().mean()  # Erreur absolue moyenne (MAE)
    
    # Coefficient de détermination R²
    y_true = results_df['TTE']
    y_pred = results_df['prediction']
    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    ss_residual = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - (ss_residual / ss_total)
    
    # Afficher les résultats des métriques avec explication
    print("\nMétriques d'évaluation du modèle:")
    
    # Explication de MSE
    print(f"\nErreur quadratique moyenne (MSE): {mse}")
    print("L'erreur quadratique moyenne (MSE) mesure la moyenne des carrés des différences entre les valeurs réelles et les valeurs prédites. Un MSE plus faible indique que le modèle a une meilleure performance de prédiction.")
    
    # Explication de RMSE
    print(f"\nRacine de l'erreur quadratique moyenne (RMSE): {rmse}")
    print("La racine de l'erreur quadratique moyenne (RMSE) est la racine carrée du MSE. Elle permet d'obtenir une mesure de l'erreur dans les mêmes unités que les valeurs de sortie du modèle. Un RMSE plus faible indique une meilleure précision du modèle.")
    
    # Explication de MAE
    print(f"\nErreur absolue moyenne (MAE): {mae}")
    print("L'erreur absolue moyenne (MAE) est la moyenne des valeurs absolues des différences entre les valeurs réelles et les valeurs prédites. Contrairement au MSE, le MAE ne pénalise pas les grandes erreurs autant.")
    
    # Explication de R²
    print(f"\nCoefficient de détermination (R²): {r2}")
    print("Le coefficient de détermination (R²) indique dans quelle mesure les variations des données réelles peuvent être expliquées par le modèle. Un R² proche de 1 signifie que le modèle explique bien la variance des données.")
    
    # Visualisation de la différence entre les valeurs réelles et les prédictions
    plt.figure(figsize=(10, 6))
    
    # Tracer les différences
    plt.plot(results_df.index, results_df['difference'], color='red', label='Différence (TTE - Prédiction)')
    
    # Tracer les valeurs réelles et prédites
    plt.plot(results_df.index, results_df['TTE'], label='Valeurs réelles (TTE)', alpha=0.6)
    plt.plot(results_df.index, results_df['prediction'], label='Valeurs prédites', alpha=0.6)
    
    # Ajouter les labels et titre
    plt.title('Valeurs Réelles vs Prédictions vs Différence')
    plt.xlabel('Index')
    plt.ylabel('Valeurs')
    plt.legend(loc='best')
    
    # Afficher le graphique
    plt.grid(True)
    plt.show()
