from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image
import io
import Generate_Caption as b
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure your Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = service_account.Credentials.from_service_account_file(
    'concise-reserve-449207-p2-3a505c75d108.json', scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

def list_images_in_folder(folder_id):
    """
    List all image files in a Google Drive folder.
    """
    query = f"'{folder_id}' in parents and mimeType contains 'image/'"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

def fetch_image_from_drive(file_id):
    """
    Fetch image content directly from Google Drive as a PIL Image.
    """
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()  # Use in-memory buffer
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    logging.info(f"Image downloaded in memory from Drive: {file_id}")
    fh.seek(0)  # Reset buffer pointer
    return Image.open(fh)  # Load image into PIL

# Folder ID of the Drive folder
DRIVE_FOLDER_ID = '1apYS55XyiM4tIZtsHvIWXqBs0bcELwSg'

# Retrieve images from the Drive folder
logging.info("Fetching image list from Drive...")
images = list_images_in_folder(DRIVE_FOLDER_ID)

# Dictionary to store captions
# Dictionary to store captions
captions = {}

# Process each image from Google Drive
for image in images:
    try:
        file_id = image['id']
        file_name = image['name']

        # Fetch the image directly from Drive
        logging.info(f"Processing image: {file_name}")
        pil_image = fetch_image_from_drive(file_id)  # This returns a PIL.Image object

        # Generate the caption using the updated generate_caption function
        caption = b.generate_caption(pil_image)  # Pass the PIL.Image object
        captions[file_name] = caption  # Save the caption with the file name as the key

        logging.info(f"Caption generated for {file_name}: {caption}")

    except Exception as e:
        logging.error(f"Error processing image {file_name}: {e}")

# Save captions to captions.json
with open('captions.json', 'w') as f:
    json.dump(captions, f, indent=4)

logging.info("Captions saved to captions.json.")