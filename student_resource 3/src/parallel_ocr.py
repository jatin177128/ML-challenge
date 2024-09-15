import cv2
import pytesseract
from concurrent.futures import ProcessPoolExecutor

# Preprocess image before OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    return gray

# OCR function to extract text from images
def extract_text(image_path):
    try:
        preprocessed_image = preprocess_image(image_path)
        return pytesseract.image_to_string(preprocessed_image)
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return ""

# Parallel OCR processing using ProcessPoolExecutor
def batch_extract_text(image_paths, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(extract_text, image_paths))

# Example usage: Extract text from all train images in parallel
train_image_paths = [os.path.join(train_image_dir, os.path.basename(url)) for url in train_image_links]
train_texts = batch_extract_text(train_image_paths)
