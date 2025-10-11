import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile
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

            if file_path.exists():
                continue

            print(f"Downloading {href}...")
            try:
                response = requests.get(file_url, stream=True)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                print(f"Failed to download {href}: {e}")

    unzip_raw_files()


def unzip_raw_files():
    """
    Extracts all .zip files in the raw data folder to subdirectories with the same base name.
    """
    for zip_path in DEST.glob('*.zip'):
        unzipped_path = zip_path.parent / zip_path.stem
        if unzipped_path.exists():
            continue

        print(f"Unzipping {zip_path.name}...")
        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_path)
        except Exception as e:
            print(f"Failed to unzip {zip_path.name}: {e}")
