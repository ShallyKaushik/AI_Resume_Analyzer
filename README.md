# 📄 AI Resume Analyzer - Intelligent Recruitment System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![PyTorch](https://img.shields.io/badge/PyTorch-Machine_Learning-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00)

## 📌 Project Overview
The **AI Resume Analyzer** is an intelligent recruitment system designed to automate and enhance the resume screening process. Traditional Applicant Tracking Systems (ATS) often rely on rigid keyword matching, which can overlook qualified candidates[cite: 1]. 

This project solves that limitation by utilizing advanced **Natural Language Processing (NLP)** and **Transformer-based machine learning models (BERT)** to understand the contextual and semantic meaning of a candidate's skills and experience[cite: 1]. 

## ✨ Key Features
* **Intelligent ATS Scoring:** Uses BERT embeddings and cosine similarity to calculate a highly accurate compatibility score between a resume and a job description[cite: 1].
* **Zero-Shot Job Classification:** Automatically categorizes uploaded resumes into specific industry domains (e.g., Software Engineering, Data Science) using Hugging Face transformer models[cite: 1].
* **Batch Processing Engine:** Upload dozens of PDF resumes simultaneously to generate a ranked leaderboard of candidates for large-scale hiring[cite: 1].
* **Interactive Data Visualization:** Features dynamic gauge charts and comparative bar graphs powered by Plotly to make data-driven hiring decisions easier[cite: 1].
* **Automated Document Parsing:** Extracts and cleans raw text directly from PDF files using PyPDF2 and NLTK text preprocessing pipelines[cite: 1].

## 🛠️ Technology Stack
* **Language:** Python 3.8+[cite: 1]
* **Frontend UI:** Streamlit[cite: 1]
* **Machine Learning:** PyTorch, Scikit-learn, Hugging Face `transformers`[cite: 1]
* **NLP & Text Processing:** NLTK, Regex[cite: 1]
* **Data Manipulation:** Pandas[cite: 1]
* **Document Extraction:** PyPDF2[cite: 1]
* **Data Visualization:** Plotly Graph Objects & Express[cite: 1]

## 🚀 Installation and Setup

Follow these instructions to run the application on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/ShallyKaushik/AI_Resume_Analyzer.git](https://github.com/ShallyKaushik/AI_Resume_Analyzer.git)
cd AI_Resume_Analyzer

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate

### 3. Install Dependencies
```bash
pip install -r requirements.txt

### 4. Run the Application
```bash
streamlit run app.py

### 💻 How to Use
##Upload Resumes: Use the file uploader to select one or multiple .pdf resume files.

##Enter Job Description: Paste the text of the job description you are hiring for into the text area.

##Analyze: Click the "Analyze Resumes & Calculate ATS Scores" button.

##Review Results:

If a single resume is uploaded, view the detailed ATS compatibility gauge chart.

If multiple resumes are uploaded, view the ranked candidate leaderboard and the comparative bar chart.

Expand the "Resume Transcripts" section to view the cleaned, preprocessed text the AI used to make its decision.

### 🔄 Development Lifecycle
This project was developed using the Iterative SDLC Model, allowing for continuous refinement of the machine learning modules, scoring mechanisms, and user interface over multiple development cycles[cite: 1].

## Made with ❤️ by Shelly Kaushik
