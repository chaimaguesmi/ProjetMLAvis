import requests
from bs4 import BeautifulSoup
import csv
import re

# Fonction pour scraper les avis d'une URL spécifique
def scrape_reviews(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Trouver tous les avis
        review_blocks = soup.find_all("li", class_="review")
        reviews = []

        for block in review_blocks:
            try:
                # Note : extraire seulement le chiffre de la note
                star_rating = block.find("div", class_="star-rating")
                rating = star_rating.get("aria-label").strip() if star_rating else "Non spécifié"
                rating = rating.split(' ')[1]  # Garder uniquement le chiffre (avant le '/')

                # Commentaire
                comment_block = block.find("div", class_="description")
                comment = comment_block.text.strip() if comment_block else "Pas de commentaire"
                
                # Ajouter les données à la liste sous forme de dictionnaire
                reviews.append({
                    "Note": rating,
                    "Commentaire": comment
                })
            except Exception as e:
                print(f"Erreur lors du traitement d'un avis : {e}")

        return reviews
    else:
        print(f"Erreur lors de l'accès à {url} : {response.status_code}")
        return []
    
def is_valid_url(line):
    pattern = re.compile(r'https?://[^\s]+')  # Détecte les URL commençant par http:// ou https://
    return bool(pattern.match(line))


# Fonction pour lire les URLs depuis un fichier CSV
def get_urls_from_csv(file_name):
    urls = []
    try:
        with open(file_name, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if is_valid_url(row["URL du produit"]):
                  urls.append(row["URL du produit"])  # Le nom de la colonne doit correspondre à celui du CSV
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
    return urls

# Fonction principale
def main():
    # Lire les URLs depuis le fichier CSV
    csv_file = "produits_category_all_pages2.csv"  # Assurez-vous que le fichier produits.csv est au même endroit que ce script
    urls = get_urls_from_csv(csv_file)
    
    # Liste pour stocker tous les avis
    all_reviews = []

    for url in urls:
        print(f"Scraping des avis pour le produit : {url}")
        reviews = scrape_reviews(url)
        all_reviews.extend(reviews)  # Ajouter les avis pour cet URL à la liste globale

    # Sauvegarder tous les avis dans un fichier CSV
    output_file = "reviews_product_category2.csv"
    if all_reviews:  # Vérifier s'il y a des avis
        with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Note", "Commentaire"], delimiter=";")
            writer.writeheader()  # Écrire les en-têtes des colonnes
            writer.writerows(all_reviews)  # Écriture de tous les avis
        print(f"Scraping terminé pour tous les produits. Les avis sont enregistrés dans '{output_file}'.")
    else:
        print("Aucun avis collecté.")

# Lancer le script principal
main()
