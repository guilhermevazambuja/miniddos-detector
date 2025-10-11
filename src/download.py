import requests
from bs4 import BeautifulSoup
from .paths import ProjectPaths

URL = "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/"
DEST = ProjectPaths.RAW_DATA_FOLDER


def download_dataset():
    """
    Downloads all files listed in the dataset directory URL and saves them to the raw data folder.
    """
    html = requests.get(URL).text
    soup = BeautifulSoup(html, 'html.parser')

    for link in soup.find_all('a'):
        href = link.get('href')
        if href and not href.startswith('?') and not href.endswith('/'):
            file_url = URL + href
            file_path = DEST / href
            print(f"Downloading {href}...")
            try:
                response = requests.get(file_url, stream=True)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                print(f"Failed to download {href}: {e}")
