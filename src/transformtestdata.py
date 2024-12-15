import pandas as pd

# Fonction pour préparer les données
def prepare_data(test_set_path, rul_test_set_path, features_cols):
    # Charger le jeu de test
    test_set = pd.read_csv(test_set_path)
    
    # Vérifier les colonnes disponibles dans test_set
    print("Colonnes dans test_set : ", test_set.columns)  # Affiche les colonnes
    
    # Garder uniquement les colonnes de caractéristiques et 'Unit number'
    test_set = test_set[features_cols + ['Unit number']]  # Utiliser 'Unit number' comme nom de colonne
    
    # Charger le fichier RUL (valeurs de TTE) pour le jeu de test
    rul_test_set = pd.read_csv(rul_test_set_path)
    
    # Afficher les colonnes de rul_test_set pour vérifier
    print("Colonnes dans rul_test_set : ", rul_test_set.columns)  # Affiche les colonnes
    
    # Extraire les valeurs de la première colonne (RUL) du fichier RUL
    rul_values = rul_test_set.iloc[:, 0].values  # Prendre les valeurs de la première colonne
    
    # Créer un dictionnaire qui mappe chaque unit_number à la valeur de RUL correspondante
    unit_number_to_rul = {i+1: rul_values[i] for i in range(len(rul_values))}
    
    # Appliquer la colonne 'RUL' à partir du dictionnaire en fonction de 'Unit number'
    test_set['RUL'] = test_set['Unit number'].map(unit_number_to_rul)
    
    # Renommer la colonne 'RUL' en 'TTE'
    test_set = test_set.rename(columns={'RUL': 'TTE'})
    
    # Supprimer la colonne 'Unit number' si vous ne souhaitez pas la conserver dans le DataFrame final
    test_set = test_set.drop(columns=['Unit number'])
    
    return test_set

if __name__ == "__main__":
    # Définir les chemins des fichiers CSV
    test_set_path = "C:/Users/Admin/Desktop/spark_project/data/test_set.csv"
    rul_test_set_path = "C:/Users/Admin/Desktop/spark_project/data/RUL_test_set.csv"
    
    # Colonnes des caractéristiques
    features_cols = ["Sensor measurement 11", "Sensor measurement 12", "Sensor measurement 13", 
                     "Sensor measurement 15", "Sensor measurement 17", "Sensor measurement 2", 
                     "Sensor measurement 20", "Sensor measurement 21", "Sensor measurement 3", 
                     "Sensor measurement 4", "Sensor measurement 7", "Sensor measurement 8"]
    
    # Préparer les données
    prepared_test_data = prepare_data(test_set_path, rul_test_set_path, features_cols)
    
    # Afficher les premières lignes pour vérifier
    print(prepared_test_data.head())
    
    # Sauvegarder le jeu de données préparé si nécessaire
    prepared_test_data.to_excel("C:/Users/Admin/Desktop/spark_project/data/prepared_test_set.xlsx", index=False)
