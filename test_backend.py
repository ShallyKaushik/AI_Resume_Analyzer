# test_backend.py
import os
from modules.pdf_extractor import extract_text_from_pdf
from modules.nlp_processor import preprocess_text

def run_tests():
    pdf_path = "data/studentChoicereport.pdf"
    
    print("--- STARTING BACKEND TESTS ---\n")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Please place '{pdf_path}' in the correct folder.")
        return

    print("✅ File found. Testing Step 1: PDF Extraction...")
    # We must open the file in binary reading mode ('rb') for PyPDF2
    with open(pdf_path, 'rb') as file:
        raw_text = extract_text_from_pdf(file)
    
    if raw_text and not raw_text.startswith("Error"):
        print("✅ Extraction Successful! Here is a snippet of the raw text:")
        print(f"   \"{raw_text[:150]}...\"\n")
    else:
        print(f"❌ Extraction Failed: {raw_text}")
        return

    print("✅ Testing Step 2: NLP Preprocessing...")
    # Testing the NLTK cleaning pipeline
    cleaned_text = preprocess_text(raw_text)
    
    if cleaned_text:
        print("✅ Preprocessing Successful! Here is the cleaned text snippet:")
        print(f"   \"{cleaned_text[:150]}...\"\n")
    else:
        print("❌ Preprocessing Failed: Returned empty string.")

if __name__ == "__main__":
    run_tests()