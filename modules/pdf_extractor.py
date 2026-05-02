# modules/pdf_extractor.py
import PyPDF2

def extract_text_from_pdf(pdf_file):
    """
    Extracts raw text from an uploaded PDF file.
    """
    text = ""
    try:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Iterate through all the pages and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"