import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# === CONFIGURATION ===
load_dotenv()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
QUERY = "lake"
NUM_IMAGES = 200
offset = 2000
root = repo_path / "bfmc_data" / "base" / "backgrounds"
OUTPUT_DIR = os.path.join(root, "bg_pixabay")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_pixabay_images(query, num_images, api_key):
    per_page = 20
    total_pages = (num_images // per_page) + 1
    images = []

    for page in range(1, total_pages + 1):
        url = (
            f"https://pixabay.com/api/?key={api_key}"
            f"&q={query.replace(' ', '+')}&image_type=photo&orientation=horizontal"
            f"&per_page={per_page}&page={page}"
        )
        response = requests.get(url)
        data = response.json()
        if "hits" in data:
            for hit in data["hits"]:
                images.append(hit["largeImageURL"])
                if len(images) >= num_images:
                    return images
    return images

downloaded_urls = set()
rep = 0
def download_images(urls, output_dir):
    global offset
    for idx, url in enumerate(tqdm(urls, desc="Downloading")):
        if url in downloaded_urls:
            rep+=1
            print("skip " + rep)
            continue  # Skip duplicates

        try:
            img_data = requests.get(url).content
            with open(os.path.join(output_dir, f"bg_{len(downloaded_urls)+offset+1}.jpg"), "wb") as f:
                f.write(img_data)
            print(f"saved bg_{len(downloaded_urls)+offset+1}.jpg")
            downloaded_urls.add(url)
        except Exception as e:
            print(f"Failed to download image {idx+offset+1}: {e}")

if __name__ == "__main__":
    urls = fetch_pixabay_images(QUERY, NUM_IMAGES, PIXABAY_API_KEY)
    download_images(urls, OUTPUT_DIR)