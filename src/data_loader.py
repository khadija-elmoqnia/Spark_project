# data_loader.py
from spark_session import create_spark_session  # Importer la fonction

def load_data(file_path, spark):
    df = spark.read.option("header", "true").csv(file_path, inferSchema=True)
    return df

if __name__ == "__main__":
    # Utilisation de la SparkSession partagée
    spark = create_spark_session()
    
    # Charger les jeux de données
    train_set = load_data("data/train_set.csv", spark)
    RUL_test_set = load_data("data/RUL_test_set.csv", spark)
    test_set = load_data("data/test_set.csv", spark)
    
    # Afficher des statistiques descriptives
    train_set.describe().show()
    train_set.describe('TTE').show()







