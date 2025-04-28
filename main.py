import streamlit as st
from openai import OpenAI
import PyPDF2
import io
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Professional Resume ATS Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #3B82F6;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1F2937;
        padding-top: 1rem;
    }
    .score-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .recommendation-item {
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 0.5rem;
        border-radius: 0 0.25rem 0.25rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client in sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Notion-logo.svg/1024px-Notion-logo.svg.png",
             width=80)
    st.markdown("### Resume ATS Analyzer")
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password")

    st.markdown("---")
    st.markdown("### Model Settings")
    model_option = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4", "gpt-3.5-turbo"],
        index=0
    )

    analysis_depth = st.slider(
        "Analysis Depth",
        min_value=1,
        max_value=5,
        value=3,
        help="Higher values provide more detailed analysis but may take longer"
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses AI to analyze your resume against ATS systems and job descriptions, 
    providing detailed feedback and recommendations for improvement.
    """)

client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to extract keywords from text
def extract_keywords(text, additional_stop_words=[]):
    stop_words = set(stopwords.words('english'))
    stop_words.update(additional_stop_words)

    # Tokenize and filter
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in word_tokens if w.isalnum() and w not in stop_words and len(w) > 2]

    # Count word frequencies
    word_freq = {}
    for word in filtered_tokens:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:50]  # Return top 50 keywords


# Generate word cloud
def generate_wordcloud(text):
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words,
        min_font_size=10,
        max_font_size=50,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


# Function to analyze resume using OpenAI API
def analyze_resume(resume_text, job_description="", depth=3):
    if not client:
        st.error("Please enter an OpenAI API key in the sidebar")
        return None

    # Adjust the prompt based on analysis depth
    detail_level = {
        1: "Provide a basic analysis with general feedback",
        2: "Provide a moderately detailed analysis",
        3: "Provide a comprehensive analysis with specific feedback for each section",
        4: "Provide a very detailed analysis with specific examples and explicit recommendations",
        5: "Provide an extremely thorough analysis with specific examples, explicit recommendations, and potential rephrasing suggestions"
    }

    prompt = f"""
    You are an expert resume analyst and career coach with 15+ years of experience in HR and recruitment.

    {detail_level[depth]}

    Analyze the following resume for ATS compatibility and provide:
    1. An overall ATS compatibility score from 0-100
    2. Detailed feedback for each section (Personal Info, Education, Experience, Skills, Projects, etc.)
    3. Specific recommendations for improvement with actionable steps
    4. Keywords analysis (missing important keywords, overused keywords, etc.)
    5. Format and structure analysis
    6. A list of the top 5-10 strengths of the resume
    7. A list of the top 5-10 areas for improvement

    If a job description is provided, analyze the resume's compatibility with the specific job requirements.

    Resume:
    {resume_text}

    Job Description:
    {job_description if job_description else "No specific job description provided"}

    Return the analysis in the following JSON format:
    {{
        "ats_score": number,
        "overall_assessment": "string",
        "format_structure": {{
            "score": number,
            "assessment": "string",
            "recommendations": "string"
        }},
        "section_feedback": [
            {{
                "section": "string",
                "score": number,
                "feedback": "string",
                "recommendations": "string"
            }}
        ],
        "keyword_analysis": {{
            "present_keywords": ["string"],
            "missing_keywords": ["string"],
            "overused_keywords": ["string"]
        }},
        "top_strengths": ["string"],
        "areas_for_improvement": ["string"],
        "general_recommendations": "string"
    }}

    IMPORTANT: Make your analysis extremely thorough, specific, and actionable. Don't be vague.
    """

    try:
        response = client.chat.completions.create(
            model=model_option,  # Use the model selected in UI
            messages=[
                {"role": "system",
                 "content": "You are a professional resume analyzer that provides detailed feedback in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
        )
        analysis_text = response.choices[0].message.content

        # Extract JSON from response (handling potential markdown code blocks)
        json_match = re.search(r'```json\n([\s\S]*?)\n```|{\s*"ats_score"[\s\S]*}', analysis_text)
        if json_match:
            analysis_json = json_match.group(1) if json_match.group(1) else json_match.group(0)
            return json.loads(analysis_json)
        else:
            return json.loads(analysis_text)
    except Exception as e:
        st.error(f"Error analyzing resume: {e}")
        return None


# UI Layout - Main content
st.markdown('<div class="main-header">Professional Resume ATS Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Boost your job application success rate with AI-powered resume analysis</div>',
            unsafe_allow_html=True)

# Main interface tabs
tab1, tab2, tab3 = st.tabs(["📝 Analyze Resume", "📊 View Results", "📚 Resume Tips"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Upload Your Resume</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file:
            # Show PDF preview
            st.success("✅ PDF uploaded successfully!")
            # Display some basic info about the PDF
            if uploaded_file.size < 1000000:  # Less than 1MB
                file_size = f"{uploaded_file.size / 1024:.1f} KB"
            else:
                file_size = f"{uploaded_file.size / (1024 * 1024):.1f} MB"
            st.markdown(f"**File name:** {uploaded_file.name}  \n**Size:** {file_size}")

            # Extract and show a preview of the text
            with st.expander("Preview extracted text"):
                text_preview = extract_text_from_pdf(uploaded_file)
                st.text_area("Extracted Content (preview)", value=text_preview[:500] + "...", height=200, disabled=True)

    with col2:
        st.markdown('<div class="section-header">Job Description (Optional)</div>', unsafe_allow_html=True)
        st.markdown("""
        Adding a job description helps the AI analyze your resume's compatibility with the specific role.
        This significantly improves the quality and relevance of recommendations.
        """)
        job_description = st.text_area("Paste the job description here", height=200)

    analyze_button = st.button("Analyze Resume", type="primary", use_container_width=True)

    if analyze_button:
        if not uploaded_file:
            st.error("Please upload a resume to analyze")
        elif not client:
            st.error("Please enter your OpenAI API key in the sidebar to continue")
        else:
            # Store the analysis in session state so it persists across tab switches
            st.session_state.resume_analyzed = True
            with st.spinner("Analyzing your resume... This may take a minute depending on the length of your resume."):
                # Extract text from the uploaded PDF
                pdf_bytes = io.BytesIO(uploaded_file.getvalue())
                resume_text = extract_text_from_pdf(pdf_bytes)

                # Show progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulate work being done
                    progress_bar.progress(i + 1)

                # Get analysis from OpenAI
                st.session_state.analysis = analyze_resume(resume_text, job_description, analysis_depth)
                st.session_state.resume_text = resume_text
                st.session_state.job_description = job_description

                # Switch to results tab after analysis
                # Fix for newer versions of Streamlit
                st.rerun()  # Changed from st.experimental_rerun()

with tab2:
    if 'resume_analyzed' in st.session_state and st.session_state.resume_analyzed:
        analysis = st.session_state.analysis
        resume_text = st.session_state.resume_text

        if analysis:
            # Overview section
            st.markdown('<div class="section-header">Resume Analysis Overview</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                # ATS Score with circular gauge
                score = analysis.get("ats_score", 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={'text': "ATS Score"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3B82F6"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "#E5E7EB"},
                            {'range': [80, 100], 'color': "#DBEAFE"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # Score interpretation
                if score < 60:
                    st.error("Your resume needs significant improvements to pass ATS systems.")
                elif score < 80:
                    st.warning("Your resume may pass some ATS systems but could be improved.")
                else:
                    st.success("Your resume is well-optimized for ATS systems!")

            with col2:
                # Format & Structure Score
                format_score = analysis.get("format_structure", {}).get("score", 0)
                st.metric("Format & Structure Score", f"{format_score}/100")

                # Average Section Score
                sections = analysis.get("section_feedback", [])
                if sections:
                    avg_section_score = sum(section.get("score", 0) for section in sections) / len(sections)
                    st.metric("Average Section Score", f"{avg_section_score:.1f}/100")

                # Keywords Analysis
                present_keywords = len(analysis.get("keyword_analysis", {}).get("present_keywords", []))
                missing_keywords = len(analysis.get("keyword_analysis", {}).get("missing_keywords", []))
                st.metric("Keywords Found/Missing", f"{present_keywords}/{missing_keywords + present_keywords}")

            with col3:
                # Word cloud from resume
                st.markdown("### Key Terms in Your Resume")
                wordcloud_fig = generate_wordcloud(resume_text)
                st.pyplot(wordcloud_fig)

            # Overall assessment
            st.markdown('<div class="section-header">Overall Assessment</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="recommendation-item">
                {analysis.get("overall_assessment", "No overall assessment provided")}
            </div>
            """, unsafe_allow_html=True)

            # Top strengths and areas for improvement
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Top Strengths")
                strengths = analysis.get("top_strengths", [])
                for i, strength in enumerate(strengths, 1):
                    st.markdown(f"**{i}.** {strength}")

            with col2:
                st.markdown("### Areas for Improvement")
                improvements = analysis.get("areas_for_improvement", [])
                for i, improvement in enumerate(improvements, 1):
                    st.markdown(f"**{i}.** {improvement}")

            # Section scores visualization
            st.markdown('<div class="section-header">Section-by-Section Analysis</div>', unsafe_allow_html=True)

            # Create DataFrame for section scores
            if sections:
                section_data = {
                    'Section': [section.get('section') for section in sections],
                    'Score': [section.get('score') for section in sections]
                }
                section_df = pd.DataFrame(section_data)

                fig = px.bar(
                    section_df,
                    x='Section',
                    y='Score',
                    color='Score',
                    color_continuous_scale=px.colors.sequential.Blues,
                    labels={'Score': 'ATS Score', 'Section': 'Resume Section'},
                    title='Section Scores Analysis'
                )
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)

            # Section feedback with expandable sections
            for section in sections:
                with st.expander(f"{section.get('section')} - Score: {section.get('score', 'N/A')}/100"):
                    st.markdown("#### Feedback")
                    st.markdown(section.get("feedback", "No feedback provided"))

                    st.markdown("#### Recommendations")
                    recs = section.get("recommendations", "No recommendations provided")
                    # Split recommendations by bullet points or newlines for better display
                    if '•' in recs:
                        rec_list = recs.split('•')
                        for rec in rec_list[1:]:  # Skip the first empty item
                            st.markdown(f"""
                            <div class="recommendation-item">
                                • {rec.strip()}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="recommendation-item">
                            {recs}
                        </div>
                        """, unsafe_allow_html=True)

            # Keywords Analysis
            st.markdown('<div class="section-header">Keywords Analysis</div>', unsafe_allow_html=True)

            kw_analysis = analysis.get("keyword_analysis", {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Present Keywords")
                present_kw = kw_analysis.get("present_keywords", [])
                if present_kw:
                    for kw in present_kw:
                        st.markdown(f"✅ {kw}")
                else:
                    st.info("No specific keywords identified")

            with col2:
                st.markdown("### Missing Keywords")
                missing_kw = kw_analysis.get("missing_keywords", [])
                if missing_kw:
                    for kw in missing_kw:
                        st.markdown(f"❌ {kw}")
                else:
                    st.success("No critical missing keywords identified")

            # Overused keywords if available
            overused_kw = kw_analysis.get("overused_keywords", [])
            if overused_kw:
                st.markdown("### Overused Keywords")
                st.markdown("These keywords appear too frequently and may be seen as keyword stuffing:")
                for kw in overused_kw:
                    st.markdown(f"⚠️ {kw}")

            # Format and structure analysis
            st.markdown('<div class="section-header">Format & Structure Analysis</div>', unsafe_allow_html=True)

            format_structure = analysis.get("format_structure", {})
            st.markdown(f"""
            <div class="recommendation-item">
                {format_structure.get("assessment", "No format analysis provided")}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Format Recommendations")
            format_recs = format_structure.get("recommendations", "No recommendations provided")
            st.markdown(f"""
            <div class="recommendation-item">
                {format_recs}
            </div>
            """, unsafe_allow_html=True)

            # General recommendations
            st.markdown('<div class="section-header">General Recommendations</div>', unsafe_allow_html=True)
            general_recs = analysis.get("general_recommendations", "No general recommendations provided")
            st.markdown(f"""
            <div class="recommendation-item">
                {general_recs}
            </div>
            """, unsafe_allow_html=True)

            # Download section
            st.markdown('<div class="section-header">Download Analysis Report</div>', unsafe_allow_html=True)

            # Create a formatted markdown report
            report_md = f"""
            # Resume Analysis Report

            ## Overall Assessment
            ATS Compatibility Score: **{analysis.get('ats_score')}/100**

            {analysis.get('overall_assessment')}

            ## Top Strengths
            {chr(10).join(['- ' + s for s in analysis.get('top_strengths', [])])}

            ## Areas for Improvement
            {chr(10).join(['- ' + s for s in analysis.get('areas_for_improvement', [])])}

            ## Section-by-Section Analysis
            """

            for section in sections:
                report_md += f"""
                ### {section.get('section')} - Score: {section.get('score')}/100

                **Feedback:**
                {section.get('feedback')}

                **Recommendations:**
                {section.get('recommendations')}
                """

            report_md += f"""
            ## Keywords Analysis

            **Present Keywords:** {', '.join(kw_analysis.get('present_keywords', []))}

            **Missing Keywords:** {', '.join(kw_analysis.get('missing_keywords', []))}

            **Overused Keywords:** {', '.join(kw_analysis.get('overused_keywords', []))}

            ## Format & Structure Analysis - Score: {format_structure.get('score')}/100

            {format_structure.get('assessment')}

            **Format Recommendations:**
            {format_structure.get('recommendations')}

            ## General Recommendations

            {analysis.get('general_recommendations')}
            """

            # Provide download button for the report
            st.download_button(
                label="Download Analysis Report",
                data=report_md,
                file_name="resume_analysis_report.md",
                mime="text/markdown",
                use_container_width=True
            )

        else:
            st.info("No analysis results available. Please analyze a resume first.")
    else:
        st.info("Please upload and analyze a resume in the 'Analyze Resume' tab.")

with tab3:
    st.markdown('<div class="section-header">Resume Writing Tips & Best Practices</div>', unsafe_allow_html=True)

    tips_expanders = [
        ("Formatting for ATS Success", """
        * Use a clean, single-column layout with standard section headings
        * Choose standard resume fonts like Arial, Calibri, or Times New Roman (10-12pt size)
        * Avoid tables, text boxes, headers/footers, and complex formatting
        * Use standard section headings (e.g., "Work Experience" instead of "Where I've Made an Impact")
        * Save your resume as a PDF to maintain formatting (unless specified otherwise)
        * Use standard file naming: FirstName-LastName-Resume.pdf
        * Keep the design simple - avoid graphics, icons, and images
        """),

        ("Keywords Optimization", """
        * Carefully read the job description and identify key skills and requirements
        * Include relevant keywords naturally throughout your resume
        * Match the exact phrasing used in the job description when possible
        * Include both spelled-out terms and acronyms (e.g., "Search Engine Optimization (SEO)")
        * Add a dedicated Skills section to list relevant technical and soft skills
        * Tailor your resume for each job application with relevant keywords
        * Use industry-standard terminology rather than company-specific terms
        """),

        ("Quantify Your Achievements", """
        * Use numbers to quantify your achievements (e.g., "Increased sales by 27%")
        * Include metrics like percentages, dollar amounts, team sizes, and timeframes
        * Focus on results and outcomes rather than just responsibilities
        * Use action verbs at the beginning of bullet points
        * Connect your actions to business impact when possible
        * Be specific about technologies, methodologies, or tools you've used
        """),

        ("Section Order & Content", """
        * Put your strongest and most relevant sections first
        * Include contact information, summary/objective, work experience, skills, and education
        * For each job, include company name, location, job title, and employment dates
        * Use bullet points for work achievements (4-6 bullets per role)
        * Focus more space on recent and relevant roles
        * Include relevant certifications, projects, or volunteer work if applicable
        * Keep your resume to 1-2 pages (unless specified otherwise for your industry)
        """),

        ("Writing Style & Grammar", """
        * Use concise, active language and avoid first-person pronouns
        * Be consistent with formatting, punctuation, and tense
        * Eliminate spelling and grammatical errors
        * Avoid jargon unless it's industry-standard terminology
        * Focus on accomplishments rather than duties
        * Remove filler words and unnecessary adjectives
        * Use parallel structure in bullet points
        """)
    ]

    for title, content in tips_expanders:
        with st.expander(title):
            st.markdown(content)

    st.markdown('<div class="section-header">Common ATS Myths</div>', unsafe_allow_html=True)

    myths = {
        "Myth: You need to 'beat' or 'trick' the ATS":
            "Reality: ATS systems aren't adversaries to overcome; they're tools to help employers manage applications. Focus on clarity and relevance rather than tricks.",

        "Myth: Stuffing keywords will improve your chances":
            "Reality: Keyword stuffing can make your resume look spammy and harm your chances when a human reviews it. Use keywords naturally and contextually.",

        "Myth: All ATS systems are the same":
            "Reality: There are many different ATS platforms with varying features and capabilities. Focus on resume best practices rather than optimizing for a specific system.",

        "Myth: Complex formatting is always rejected by ATS":
            "Reality: Modern ATS systems have improved at parsing formatting, but it's still best to keep formatting simple to ensure compatibility with all systems.",

        "Myth: If you're rejected, it's always because of the ATS":
            "Reality: There are many reasons for rejection beyond ATS issues, including qualification mismatch, experience level, or simply high competition."
    }

    for myth, reality in myths.items():
        st.markdown(f"**{myth}**")
        st.markdown(f"_{reality}_")
        st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    This AI-powered tool provides analysis based on general best practices and may not account for industry-specific standards or unique situations. 
    Always use your judgment and seek professional advice when needed.
</div>
""", unsafe_allow_html=True)