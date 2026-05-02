# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.pdf_extractor import extract_text_from_pdf
from modules.nlp_processor import preprocess_text
from modules.ai_matcher import calculate_ats_score, classify_resume

# Page Configuration
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")
st.subheader("Intelligent Recruitment System")
st.markdown("---")

# Layout: Two main columns (Left for inputs, Right for results)
col_input, col_results = st.columns([1, 1])

with col_input:
    st.write("### 1. Upload Candidate Resumes")
    # File uploader set to accept multiple files
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    st.write("### 2. Enter Job Description")
    job_description = st.text_area("Paste the job description here to compare:", height=200)
    
    analyze_btn = st.button("Analyze Resumes & Calculate ATS Scores", type="primary", use_container_width=True)

with col_results:
    st.write("### Analysis Results")
    
    if analyze_btn:
        if not uploaded_files:
            st.error("Please upload at least one resume.")
        elif not job_description.strip():
            st.error("Please enter a job description.")
        else:
            clean_jd = preprocess_text(job_description)
            results_data = [] # List to store data for our Pandas DataFrame
            
            # Progress bar for batch processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})...")
                
                # Extract and clean text
                raw_text = extract_text_from_pdf(file)
                clean_resume = preprocess_text(raw_text)
                
                # AI Scoring and Classification
                ats_score = calculate_ats_score(clean_resume, clean_jd)
                category, confidence = classify_resume(clean_resume)
                
                # Append to our results list
                results_data.append({
                    "Candidate File": file.name,
                    "ATS Score (%)": ats_score,
                    "Predicted Domain": category,
                    "Raw Text": clean_resume # Storing for the expander later
                })
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.empty() # Clear the status text when done
            st.success(f"Batch Analysis Complete for {len(uploaded_files)} candidate(s)!")
            
            # Create a Pandas DataFrame and sort by top ATS Score
            df = pd.DataFrame(results_data)
            df = df.sort_values(by="ATS Score (%)", ascending=False).reset_index(drop=True)
            
            # 1. Display the Ranked Leaderboard Table
            st.write("#### Candidate Leaderboard")
            
            # Displaying dataframe without the Raw Text column to keep it clean
            display_df = df[["Candidate File", "ATS Score (%)", "Predicted Domain"]]
            st.dataframe(
                display_df.style.background_gradient(subset=['ATS Score (%)'], cmap='Greens'),
                use_container_width=True
            )
            
            # 2. Display Charts depending on number of files uploaded
            if len(df) > 1:
                st.write("#### ATS Score Comparison")
                fig = px.bar(df, x='Candidate File', y='ATS Score (%)', color='ATS Score (%)', 
                             color_continuous_scale='Greens', text_auto='.2f')
                fig.update_layout(xaxis_title="", yaxis_title="Match Percentage", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                
            elif len(df) == 1:
                # If only one file, show the detailed gauge chart
                single_score = df.iloc[0]["ATS Score (%)"]
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = single_score,
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
                st.plotly_chart(fig, use_container_width=True)
                
                # Simple interpretation logic for a single file
                if single_score >= 80:
                    st.success("High Match: This candidate is a strong fit for the role.")
                elif single_score >= 50:
                    st.warning("Medium Match: The candidate meets some requirements, but may lack specific keywords or skills.")
                else:
                    st.error("Low Match: This candidate's profile does not align well with the job description.")
            
            # 3. View Processed Text Expander for all candidates
            st.write("#### Resume Transcripts")
            for index, row in df.iterrows():
                with st.expander(f"View Processed Text: {row['Candidate File']}"):
                    st.write(f"**Predicted Domain:** {row['Predicted Domain']}")
                    st.write(row['Raw Text'])