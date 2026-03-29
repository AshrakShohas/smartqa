"""
extractor.py
Extracts plain text from PDF, DOCX, PPTX, IMAGES, and TXT files.
"""

import io
import fitz          # PyMuPDF
import docx          # python-docx
from pptx import Presentation
from image_processor import extract_text_from_image, extract_text_from_txt, extract_page_references


def extract_text_from_file(uploaded_file, chunk_size: int = 20):
    """
    Accepts a Streamlit UploadedFile object.
    Returns (extracted_text, metadata) as a tuple.
    """
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    uploaded_file.seek(0)
    
    if name.endswith(".pdf"):
        return _extract_pdf(data, name)
    elif name.endswith(".docx"):
        return _extract_docx(data, name)
    elif name.endswith((".pptx", ".ppt")):
        return _extract_pptx(data, name)
    elif name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        return _extract_image(data, name)
    elif name.endswith(".txt"):
        return _extract_txt(data, name)
    else:
        return "", {}


def _extract_pdf(data: bytes, filename: str) -> tuple[str, dict]:
    """Extract text from PDF with page numbers."""
    text_parts = []
    metadata = {
        "source_file": filename,
        "pages": [],
        "formulas": [],
        "sections": [],
        "images": []  # Store image references
    }
    
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")
                
                # Extract page-specific metadata
                page_ref = extract_page_references(page_text, filename, page_num)
                metadata["pages"].append(page_num)
                metadata["formulas"].extend(page_ref["formulas"])
                metadata["sections"].extend(page_ref["sections"])
                
                # Check if page has images
                images = page.get_images()
                if images:
                    metadata["images"].append({
                        "page": page_num,
                        "count": len(images)
                    })
        
        doc.close()
        
        # Remove duplicate formulas
        metadata["formulas"] = list(dict.fromkeys(metadata["formulas"]))
        metadata["sections"] = list(dict.fromkeys(metadata["sections"]))
        
    except Exception as e:
        text_parts.append(f"[PDF extraction error: {e}]")
    
    return "\n".join(text_parts), metadata


def _extract_docx(data: bytes, filename: str) -> tuple[str, dict]:
    """Extract text from DOCX with section info."""
    text_parts = []
    metadata = {
        "source_file": filename,
        "pages": [1],  # DOCX doesn't have pages, use 1
        "formulas": [],
        "sections": [],
        "images": []
    }
    
    try:
        doc = docx.Document(io.BytesIO(data))
        section_counter = 1
        current_section = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                # Detect section headers (often bold or larger font)
                if para.style.name and 'Heading' in para.style.name:
                    if current_section:
                        text_parts.append(f"[Section {section_counter}]\n" + "\n".join(current_section))
                        section_counter += 1
                        current_section = []
                    metadata["sections"].append(para.text.strip())
                else:
                    current_section.append(para.text)
                
                text_parts.append(para.text)
        
        # Add last section
        if current_section:
            text_parts.append(f"[Section {section_counter}]\n" + "\n".join(current_section))
        
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)
        
        # Detect formulas from all text
        full_text = "\n".join(text_parts)
        from image_processor import detect_formulas
        metadata["formulas"] = detect_formulas(full_text)
        
    except Exception as e:
        text_parts.append(f"[DOCX extraction error: {e}]")
    
    return "\n".join(text_parts), metadata


def _extract_pptx(data: bytes, filename: str) -> tuple[str, dict]:
    """Extract text from PPTX with slide numbers."""
    text_parts = []
    metadata = {
        "source_file": filename,
        "pages": [],
        "formulas": [],
        "sections": [],
        "images": []
    }
    
    try:
        prs = Presentation(io.BytesIO(data))
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_texts.append(shape.text.strip())
            
            if slide_texts:
                text_parts.append(f"[Slide {slide_num}]\n" + "\n".join(slide_texts))
                metadata["pages"].append(slide_num)
                
                # Check for images in slide
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # Picture type
                        metadata["images"].append({
                            "slide": slide_num,
                            "has_image": True
                        })
        
        # Detect formulas
        full_text = "\n".join(text_parts)
        from image_processor import detect_formulas
        metadata["formulas"] = detect_formulas(full_text)
        
    except Exception as e:
        text_parts.append(f"[PPTX extraction error: {e}]")
    
    return "\n".join(text_parts), metadata


def _extract_image(data: bytes, filename: str) -> tuple[str, dict]:
    """Extract text from image using OCR."""
    text, formulas = extract_text_from_image(data)
    metadata = {
        "source_file": filename,
        "pages": [1],
        "formulas": formulas,
        "sections": [],
        "images": [{"original": True}],
        "is_image": True
    }
    
    # Add image for display (convert to base64)
    import base64
    image_base64 = base64.b64encode(data).decode('utf-8')
    metadata["image_data"] = f"data:image/{filename.split('.')[-1]};base64,{image_base64}"
    
    return text, metadata


def _extract_txt(data: bytes, filename: str) -> tuple[str, dict]:
    """Extract text from TXT file."""
    text = extract_text_from_txt(data)
    metadata = {
        "source_file": filename,
        "pages": [1],
        "formulas": [],
        "sections": [],
        "is_text": True
    }
    
    # Detect formulas in text
    from image_processor import detect_formulas
    metadata["formulas"] = detect_formulas(text)
    
    return text, metadata