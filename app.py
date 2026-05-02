# app.py
import streamlit as st
from modules.pdf_extractor import extract_text_from_pdf
from modules.nlp_processor import preprocess_text
from modules.ai_matcher import calculate_ats_score
from modules.ai_matcher import calculate_ats_score, classify_resume
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")
st.subheader("Intelligent Recruitment System")
st.markdown("---")

# Layout: Two main columns (Left for inputs, Right for results)
col_input, col_results = st.columns([1, 1])

with col_input:
    st.write("### 1. Upload Candidate Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    st.write("### 2. Enter Job Description")
    job_description = st.text_area("Paste the job description here to compare:", height=200)
    
    analyze_btn = st.button("Analyze Resume & Calculate ATS Score", type="primary", use_container_width=True)

with col_results:
    st.write("### Analysis Results")
    
    if analyze_btn:
        if uploaded_file is None:
            st.error("Please upload a resume first.")
        elif not job_description.strip():
            st.error("Please enter a job description.")
        else:
            with st.spinner("Processing Document..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                clean_resume = preprocess_text(raw_text)
                clean_jd = preprocess_text(job_description)
            

            with st.spinner("Analyzing semantics with BERT... (This may take a moment)"):
                     ats_score = calculate_ats_score(clean_resume, clean_jd)
            # Add the classification call here
            category, confidence = classify_resume(clean_resume)
            
        # Displaying the Results
        st.success("Analysis Complete!")
        
        # Create a row of metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="ATS Compatibility Score", value=f"{ats_score:.2f}%")
        with metric_col2:
            st.metric(label="Predicted Domain", value=category, delta=f"{confidence:.1f}% confidence")
            
        # Visual progress bar for the ATS score
        # Visual progress bar for the ATS score (REPLACE your old st.progress line with this)
        
        # Create an interactive Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = ats_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Match Compatibility", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "white"},
                'bgcolor': "black",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}],
            }
        ))
        
        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        # Simple interpretation logic
        if ats_score >= 80:
            st.success("High Match: This candidate is a strong fit for the role.")
        elif ats_score >= 50:
            st.warning("Medium Match: The candidate meets some requirements, but may lack specific keywords or skills.")
        else:
            st.error("Low Match: This candidate's profile does not align well with the job description.")
            
        with st.expander("View Processed Resume Text"):
            st.write(clean_resume)