import requests
from bs4 import BeautifulSoup
import csv

# Entrée de l'utilisateur
produit = input("Donner le nom du produit : ")

# URL de base
base_url = f"https://parapharmacie.tn/product-category/{produit}"
print(f"URL de base : {base_url}")

# Fonction principale
def main():
    page_number = 1
    produits_details = []  # Liste pour stocker toutes les URLs des produits

    while True:
        # Construire l'URL de la page actuelle
        url = f"{base_url}/page/{page_number}"
        print(f"Traitement de la page : {url}")
        page = requests.get(url)

        # Vérifier si la page existe (statut HTTP)
        if page.status_code != 200:
            print("Fin de la pagination ou erreur d'accès à la page.")
            break

        # Charger le contenu de la page
        src = page.content
        soup = BeautifulSoup(src, 'lxml')

        # Trouver tous les produits
        produits = soup.find_all("a", {'class': 'woocommerce-LoopProduct-link woocommerce-loop-product__link'})

        # Si aucun produit n'est trouvé, on arrête la boucle
        if not produits:
            print("Aucun produit trouvé sur cette page. Fin de la pagination.")
            break

        # Extraire les URLs en vérifiant les doublons
        for produit in produits:
            href = produit.get('href')
            if {"URL du produit": href} not in produits_details:
                produits_details.append({"URL du produit": href})

        # Passer à la page suivante
        page_number += 1

    # Écriture dans un fichier CSV
    if produits_details:  # Vérifier si des produits ont été trouvés
        keys = produits_details[0].keys()
        with open('produits_category_all_pages2.csv', 'a', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(produits_details)
        print("Fichier CSV généré : produits_category_all_pages2.csv")
    else:
        print("Aucun produit n'a été trouvé pour cette catégorie.")

# Appel de la fonction principale
main()
