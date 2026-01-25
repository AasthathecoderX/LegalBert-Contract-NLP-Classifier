# ============================================================================
# FILE: backend/utils/pdf_extractor.py
# ============================================================================
"""
PDF text extraction utilities
"""

import PyPDF2
import re

class PDFTextExtractor:
    """Extract text from PDF files"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('\x00', '')
            return text.strip()
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
