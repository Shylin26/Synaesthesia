import os
import zipfile

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'ravdess')

def download_ravdess():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, 'ravdess.zip')

    if not os.path.exists(zip_path):
        print(f"ZIP not found at {zip_path}")
        print("Please download it manually from:")
        print(f"  {RAVDESS_URL}")
        print(f"Save it to: {zip_path}")
        return

    print("ZIP found. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR}")

if __name__ == "__main__":
    download_ravdess()
