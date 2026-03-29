"""
image_processor.py
Handles image extraction and OCR for images and documents.
"""

import io
import base64
from PIL import Image
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import streamlit as st

# Try to configure tesseract path (adjust for your system)
# For Windows, you might need to set the path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image_data: bytes) -> tuple[str, list]:
    """
    Extract text from image using OCR and also return image for display.
    Returns: (extracted_text, list_of_image_paths_for_display)
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image for better OCR
        # Convert to numpy array for OpenCV
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Optional: Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(denoised)
        
        # Extract text with OCR
        text = pytesseract.image_to_string(processed_image, lang='eng+ben')  # Add Bengali support
        
        # Also try to detect formulas (simple pattern detection)
        formulas = detect_formulas(text)
        
        return text.strip(), formulas
        
    except Exception as e:
        return f"[Image extraction error: {e}]", []


def detect_formulas(text: str) -> list:
    """
    Detect mathematical formulas and equations in text.
    """
    import re
    
    formulas = []
    
    # Common formula patterns
    patterns = [
        r'[A-Za-z]\s*=\s*[A-Za-z0-9+\-*/^()]+',  # Simple equations
        r'\bE\s*=\s*mc\^?2?\b',  # Einstein's equation
        r'\bF\s*=\s*ma\b',  # Force
        r'\bPV\s*=\s*nRT\b',  # Ideal gas
        r'\bΔG\s*=\s*ΔH\s*-\s*TΔS\b',  # Gibbs free energy
        r'\b[a-z]+\s*=\s*[a-z]+\s*[+\-*/]\s*[a-z]+\b',  # General formulas
        r'∑|∫|∏|√|π|θ|Δ|α|β|γ',  # Math symbols
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        formulas.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_formulas = []
    for f in formulas:
        if f not in seen:
            seen.add(f)
            unique_formulas.append(f)
    
    return unique_formulas[:10]  # Return top 10 formulas


def extract_text_from_txt(data: bytes) -> str:
    """
    Extract text from TXT file.
    """
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return data.decode('latin-1')
        except:
            return "[TXT extraction error: Could not decode file]"


def extract_page_references(text: str, source_file: str, page_num: int = None) -> dict:
    """
    Extract references to specific pages or sections.
    """
    references = {
        "source_file": source_file,
        "page": page_num,
        "sections": [],
        "formulas": detect_formulas(text)
    }
    
    # Detect section headers (common patterns)
    import re
    section_patterns = [
        r'^(Chapter|Section|§)\s+\d+',  # Chapter/Section numbers
        r'^\d+\.\d+\s+[A-Z]',  # Numbered sections
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title case headers
    ]
    
    lines = text.split('\n')
    for line in lines[:20]:  # Check first 20 lines for sections
        for pattern in section_patterns:
            if re.match(pattern, line.strip()):
                references["sections"].append(line.strip())
    
    return references