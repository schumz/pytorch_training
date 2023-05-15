import pandas as pd

# Liste d'exemple
ma_liste = ['a', 'b', 'c', 'd']

# Fonction qui génère 5 éléments pour chaque élément de la liste
def generer_elements(element):
    return [element + str(i) for i in range(1, 6)]

# Créez une liste pour stocker les données du dataframe
donnees_df = []

# Itérez sur la liste et générez 5 éléments pour chaque élément de la liste
for element in ma_liste:
    elements_generees = generer_elements(element)
    donnees_df.append([element] + elements_generees)

# Créez un dataframe avec les données
df = pd.DataFrame(donnees_df)

# Affichez le dataframe
print(df)
