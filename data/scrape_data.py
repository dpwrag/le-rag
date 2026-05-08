import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

url = "https://www.wongnai.com/businesses?regions=18986&cregion=9681&categoryGroupId=9&categories=3&categories=6&categories=8&categories=10&categories=13&categories=34&categories=2&categories=49&categories=9&categories=40&categories=41&categories=44&categories=4&categories=5&categories=55&categories=33&categories=7&features.bookable=true&mode=2&page.number=1&page.size=50&rerank=false&domain=1"

headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

restaurant_links = set()

for a in soup.find_all("a", href=True):
    href = a["href"]

    if "/restaurants/" in href:
        if "reviews" in href:
            continue

        full_url = urljoin("https://www.wongnai.com", href)
        clean_url = full_url.split("?")[0].split("#")[0]
        restaurant_links.add(clean_url)

print(f"Found {len(restaurant_links)} unique restaurant links\n")

for link in sorted(restaurant_links):
    print(link)

with open("restaurant_links.txt", "w", encoding="utf-8") as f:
    for link in sorted(restaurant_links):
        f.write(link + "\n")

print("\nSaved to restaurant_links.txt")
