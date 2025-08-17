import streamlit as st
import pandas as pd
import json
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import PyPDF2
import io
from dataclasses import dataclass
import re
from dotenv import load_dotenv
from pathlib import Path

# LangGraph + Groq Integration
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from groq import Groq

# Pydantic for data validation
from pydantic import BaseModel, Field, field_validator

# Load environment variables
load_dotenv()

# Pydantic Models for Data Validation
class CVAnalysisResult(BaseModel):
    score: float = Field(..., ge=0, le=100)
    skills_match: float = Field(..., ge=0, le=100)
    rationale: str = Field(..., max_length=1000)
    highlights: List[str] = Field(default_factory=list, max_items=5)
    red_flags: List[str] = Field(default_factory=list, max_items=5)
    recommendation: str = Field(..., pattern="^(Proceed to Interview|Consider with Caution|Not Recommended)$")
    candidate_name: Optional[str] = Field(None, max_length=100)

class RobustCVState(TypedDict):
    cv_text: str
    job_title: str
    job_desc: str
    min_req: str
    score: Optional[float]
    rationale: Optional[str]
    highlights: Optional[List[str]]
    red_flags: Optional[List[str]]
    recommendation: Optional[str]
    skills_match: Optional[float]
    error: Optional[str]
    candidate_name: Optional[str]
    phone_number: Optional[str]
    email: Optional[str]

# Data Models
@dataclass
class CVMetadata:
    id: str
    filename: str
    upload_ts: datetime
    raw_text: str
    score: Optional[float] = None
    rationale: Optional[str] = None
    highlights: Optional[List[str]] = None
    red_flags: Optional[List[str]] = None
    recommendation: Optional[str] = None
    skills_match: Optional[float] = None
    candidate_name: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None

@dataclass
class JobRecord:
    id: str
    title: str
    jobdesc: str
    min_requirements: str
    created_ts: datetime

# Configure page
st.set_page_config(
    page_title="HR ATS - CV Screening System",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
def initialize_session_state():
    if 'cvs' not in st.session_state:
        st.session_state.cvs = {}
    if 'current_job' not in st.session_state:
        st.session_state.current_job = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

# API Configuration
def setup_groq_api():
    """Setup Groq API with environment variable or sidebar input"""
    try:
        # Check if API key exists in environment variables
        groq_api_key = os.environ.get("GROQ_API_KEY")
        
        if groq_api_key:
            # Use environment variable
            try:
                client = Groq(api_key=groq_api_key)
                test_response = client.chat.completions.create(
                    model="moonshotai/kimi-k2-instruct",
                    messages=[{"role": "user", "content": "Test connection"}],
                    max_tokens=10
                )
                st.sidebar.success("✅ Groq API connected (from environment)")
                return True, client
            except Exception as e:
                st.sidebar.error(f"❌ Invalid GROQ_API_KEY in environment: {str(e)}")
                return False, None
        else:
            # Show sidebar input if no environment variable
            groq_api_key = st.sidebar.text_input(
                "🔑 Groq API Key", 
                type="password",
                help="Enter your Groq API key to enable AI analysis",
                placeholder="gsk_..."
            )
            
            if not groq_api_key:
                st.sidebar.warning("⚠️ Please enter your Groq API key to continue")
                return False, None
            
            # Test the API key from sidebar
            try:
                client = Groq(api_key=groq_api_key)
                test_response = client.chat.completions.create(
                    model="moonshotai/kimi-k2-instruct",
                    messages=[{"role": "user", "content": "Test connection"}],
                    max_tokens=10
                )
                st.sidebar.success("✅ Groq API connection successful")
                return True, client
            except Exception as e:
                st.sidebar.error(f"❌ Invalid API key: {str(e)}")
                return False, None
            
    except Exception as e:
        st.sidebar.error(f"❌ API setup error: {str(e)}")
        return False, None

# PDF Text Extraction
def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def preprocess_text(text: str) -> str:
    """Clean and normalize extracted text"""
    text = " ".join(text.split())
    text = re.sub(r'[^\w\s\.\,\:\;\-\(\)@]', '', text)
    return text[:4000]

# Contact Information Extraction
def extract_contact_info(cv_text: str) -> Dict[str, Optional[str]]:
    """Extract contact information from CV text"""
    contact_info = {"name": None, "phone": None, "email": None}
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    email_matches = re.findall(email_pattern, cv_text)
    if email_matches:
        valid_emails = [email for email in email_matches 
                       if not email.startswith('.') and '..' not in email]
        if valid_emails:
            contact_info["email"] = valid_emails[0]
    
    # Extract phone number
    phone_patterns = [
        r'(?:\+?62|0)[\s-]?(?:\d{2,4})[\s-]?\d{3,4}[\s-]?\d{3,5}',
        r'(?:\+?1)?[\s-]?\(?(\d{3})\)?[\s-]?(\d{3})[\s-]?(\d{4})',
        r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b'
    ]
    
    for pattern in phone_patterns:
        phone_match = re.search(pattern, cv_text, re.IGNORECASE)
        if phone_match:
            contact_info["phone"] = phone_match.group(0).strip()
            break
    
    # Extract name (first 500 characters)
    cv_start = cv_text[:500]
    name_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'(?:Name|Nama)[\s:]*([A-Za-z\s]+?)(?:\n|$|[,.])'
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, cv_start, re.MULTILINE | re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            if 2 <= len(name.split()) <= 5:
                contact_info["name"] = name
                break
    
    return contact_info

# Robust JSON Parsing
def safe_json_parse(response_text: str) -> dict:
    """Safely parse JSON with fallback strategies"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Extract from markdown
    try:
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0]
            return json.loads(json_content)
    except:
        pass
    
    # Manual extraction fallback
    try:
        result = {}
        patterns = {
            'score': r'"score":\s*(\d+(?:\.\d+)?)',
            'skills_match': r'"skills_match":\s*(\d+(?:\.\d+)?)',
            'rationale': r'"rationale":\s*"([^"]*)"',
            'recommendation': r'"recommendation":\s*"([^"]*)"'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response_text)
            if match:
                if key in ['score', 'skills_match']:
                    result[key] = float(match.group(1))
                else:
                    result[key] = match.group(1)
        
        if result:
            return result
    except:
        pass
    
    return {
        "score": 0.0,
        "skills_match": 0.0,
        "rationale": "AI response parsing failed",
        "highlights": ["Processing error"],
        "red_flags": ["Manual review required"],
        "recommendation": "Manual Review Required"
    }

# LangGraph Workflow
def create_robust_cv_scoring_graph():
    """Create robust CV scoring workflow"""
    builder = StateGraph(RobustCVState)
    
    def preprocess_node(state: RobustCVState) -> RobustCVState:
        try:
            state["cv_text"] = preprocess_text(state["cv_text"])
            return state
        except Exception as e:
            state["error"] = f"Preprocessing error: {str(e)}"
            return state
    
    def scoring_node(state: RobustCVState) -> RobustCVState:
        if state.get("error"):
            return state
            
        try:
            # Extract contact information
            contact_info = extract_contact_info(state["cv_text"])
            state["candidate_name"] = contact_info["name"]
            state["phone_number"] = contact_info["phone"]
            state["email"] = contact_info["email"]
            
            client = st.session_state.groq_client
            
            prompt = f"""
            You are an expert HR recruiter. Analyze this CV for the job position and respond with ONLY valid JSON format.

            JOB POSITION: {state['job_title']}
            JOB DESCRIPTION: {state['job_desc']}
            MINIMUM REQUIREMENTS: {state['min_req']}
            
            CANDIDATE CV TEXT:
            {state['cv_text']}

            Provide your analysis in this exact JSON format:
            {{
                "score": [number between 0-100],
                "skills_match": [percentage 0-100 of required skills matched],
                "rationale": "[2-3 sentences explaining the score]",
                "highlights": ["key strength 1", "key strength 2", "key strength 3"],
                "red_flags": ["concern 1", "concern 2"] or [],
                "recommendation": "Proceed to Interview" or "Consider with Caution" or "Not Recommended",
                "candidate_name": "[full name if clearly found in CV, otherwise null]"
            }}

            Base your scoring on:
            - Requirements match (40%)
            - Experience relevance (30%) 
            - Skills alignment (20%)
            - Overall fit (10%)
            """

            response = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            result = safe_json_parse(response_text)
            
            # Populate state
            state["score"] = max(0, min(100, float(result.get("score", 0))))
            state["skills_match"] = max(0, min(100, float(result.get("skills_match", 0))))
            state["rationale"] = result.get("rationale", "Analysis completed")
            state["highlights"] = result.get("highlights", [])
            state["red_flags"] = result.get("red_flags", [])
            state["recommendation"] = result.get("recommendation", "Manual Review Required")
            
            # Use AI name if better than regex extraction
            ai_name = result.get("candidate_name")
            if ai_name and not state["candidate_name"]:
                state["candidate_name"] = ai_name
            
        except Exception as e:
            state["error"] = f"AI Analysis Error: {str(e)}"
            state["score"] = 0.0
            state["skills_match"] = 0.0
            state["rationale"] = f"Error: {str(e)}"
            state["highlights"] = ["Processing error"]
            state["red_flags"] = ["Manual review required"]
            state["recommendation"] = "Manual Review Required"
        
        return state
    
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("score", scoring_node)
    
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "score")
    builder.add_edge("score", END)
    
    return builder.compile()

def score_cv_with_groq(cv_text: str, job_title: str, job_desc: str, min_req: str) -> Dict[str, Any]:
    """Score CV using Groq AI with robust error handling"""
    try:
        workflow = create_robust_cv_scoring_graph()
        
        initial_state = RobustCVState(
            cv_text=cv_text,
            job_title=job_title,
            job_desc=job_desc,
            min_req=min_req,
            score=None,
            rationale=None,
            highlights=None,
            red_flags=None,
            recommendation=None,
            skills_match=None,
            error=None,
            candidate_name=None,
            phone_number=None,
            email=None
        )
        
        final_state = workflow.invoke(initial_state)
        
        return {
            "score": final_state.get("score", 0.0),
            "skills_match": final_state.get("skills_match", 0.0),
            "rationale": final_state.get("rationale", "Analysis completed"),
            "highlights": final_state.get("highlights", []),
            "red_flags": final_state.get("red_flags", []),
            "recommendation": final_state.get("recommendation", "Manual Review Required"),
            "candidate_name": final_state.get("candidate_name"),
            "phone_number": final_state.get("phone_number"),
            "email": final_state.get("email")
        }
        
    except Exception as e:
        contact_info = extract_contact_info(cv_text)
        return {
            "score": 0.0,
            "skills_match": 0.0,
            "rationale": f"Workflow Error: {str(e)}",
            "highlights": ["Processing error"],
            "red_flags": ["Manual review required"],
            "recommendation": "Manual Review Required",
            "candidate_name": contact_info["name"],
            "phone_number": contact_info["phone"],
            "email": contact_info["email"]
        }

def main():
    initialize_session_state()
    
    st.title("🎯 HR ATS - CV Screening System")    
    # Setup Groq API in sidebar
    api_configured, groq_client = setup_groq_api()
    
    if not api_configured:
        st.error("🔑 Please configure your Groq API key in the sidebar to continue!")
        st.info("💡 Get your API key from [Groq Console](https://console.groq.com/)")
        st.stop()
    
    # Store client in session state
    st.session_state.groq_client = groq_client
    
    cv_analyzer_page()

def cv_analyzer_page():
    """Main CV analysis page"""
    
    if not st.session_state.analysis_complete:
        st.header("📋 Job Setup & CV Analysis")        
        with st.form("job_analysis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
            
            with col2:
                job_desc = st.text_area("Job Description", height=100, 
                                       placeholder="Describe role, responsibilities, key qualifications...")
                min_requirements = st.text_area("Minimum Requirements", height=100,
                                               placeholder="Essential skills, experience, education...")
            
            st.subheader("📄 Upload CV Files")
            uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], 
                                            accept_multiple_files=True,
                                            help="Upload multiple CV files in PDF format")
            
            analyze_button = st.form_submit_button("🚀 Analyze CVs", type="primary")
            
            if analyze_button:
                if not all([job_title, job_desc, min_requirements]):
                    st.error("Please fill in all job requirement fields!")
                    return
                
                if not uploaded_files:
                    st.error("Please upload at least one CV file!")
                    return
                
                # Create job record
                st.session_state.current_job = JobRecord(
                    id=str(uuid.uuid4()),
                    title=job_title,
                    jobdesc=job_desc,
                    min_requirements=min_requirements,
                    created_ts=datetime.now()
                )
                
                process_and_analyze_cvs(uploaded_files, job_title, job_desc, min_requirements)
    
    else:
        display_analysis_results()

def process_and_analyze_cvs(uploaded_files, job_title, job_desc, min_requirements):
    """Process and analyze CVs"""
    st.subheader("⚙️ Processing CVs...")
    
    st.session_state.cvs.clear()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Analyzing {uploaded_file.name}... ({i+1}/{total_files})")
        
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        if pdf_text:
            clean_text = preprocess_text(pdf_text)
            
            cv_id = str(uuid.uuid4())
            cv_metadata = CVMetadata(
                id=cv_id,
                filename=uploaded_file.name,
                upload_ts=datetime.now(),
                raw_text=clean_text
            )
            
            try:
                result = score_cv_with_groq(clean_text, job_title, job_desc, min_requirements)
                
                cv_metadata.score = result["score"]
                cv_metadata.skills_match = result["skills_match"]
                cv_metadata.rationale = result["rationale"]
                cv_metadata.highlights = result["highlights"]
                cv_metadata.red_flags = result["red_flags"]
                cv_metadata.recommendation = result["recommendation"]
                cv_metadata.candidate_name = result.get("candidate_name")
                cv_metadata.phone_number = result.get("phone_number")
                cv_metadata.email = result.get("email")
                
            except Exception as e:
                cv_metadata.score = 0
                cv_metadata.rationale = f"Processing error: {str(e)}"
            
            st.session_state.cvs[cv_id] = cv_metadata
        
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text("✅ Analysis Complete!")
    st.session_state.analysis_complete = True
    st.rerun()

def display_analysis_results():
    """Display analysis results"""
    job = st.session_state.current_job
    
    st.header("📊 Analysis Results")
    
    # Summary stats
    total_cvs = len(st.session_state.cvs)
    cvs_list = list(st.session_state.cvs.values())
    avg_score = sum([cv.score for cv in cvs_list if cv.score]) / total_cvs if total_cvs else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 CVs Analyzed", total_cvs)
    with col2:
        st.metric("📈 Average Score", f"{avg_score:.1f}/100")
    with col3:
        processing_time = (datetime.now() - job.created_ts).seconds
        st.metric("⚡ Processing Time", f"{processing_time//60}m {processing_time%60}s")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Results"):
            export_results()
    with col2:
        if st.button("🆕 New Analysis"):
            reset_analysis()
            st.rerun()
    
    # Display all candidates sorted by score
    st.subheader("🏆 All Candidates (Ranked by Score)")
    
    sorted_cvs = sorted(st.session_state.cvs.values(), key=lambda x: x.score if x.score else 0, reverse=True)
    
    for i, cv in enumerate(sorted_cvs, 1):
        with st.container():
            # Score color coding
            if cv.score >= 80:
                score_color = "🟢"
            elif cv.score >= 60:
                score_color = "🟡"
            else:
                score_color = "🔴"
            
            col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 1])
            
            with col1:
                st.write(f"**#{i}**")
            with col2:
                display_name = cv.candidate_name if cv.candidate_name else cv.filename
                st.write(f"**{display_name}**")
                if cv.candidate_name and cv.filename != cv.candidate_name:
                    st.caption(f"File: {cv.filename}")
            with col3:
                st.write(f"{score_color} Score: {cv.score:.1f}/100")

            
            # Detailed analysis
            with st.expander(f"📋 Detailed Analysis - {display_name}"):
                # Contact info
                st.write("**📞 Contact Information:**")
                contact_col1, contact_col2, contact_col3 = st.columns(3)
                
                with contact_col1:
                    st.write(f"**Name:** {cv.candidate_name or 'Not found'}")
                with contact_col2:
                    st.write(f"**Phone:** {cv.phone_number or 'Not found'}")
                with contact_col3:
                    st.write(f"**Email:** {cv.email or 'Not found'}")
                
                st.divider()
                
                # Analysis details
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.write("**📊 Scores:**")
                    st.write(f"Overall: {cv.score:.1f}/100")
                    st.write("**💡 Recommendation:**")
                    
                    if cv.recommendation == "Proceed to Interview":
                        st.success(f"✅ {cv.recommendation}")
                    elif cv.recommendation == "Consider with Caution":
                        st.warning(f"⚠️ {cv.recommendation}")
                    else:
                        st.error(f"❌ {cv.recommendation}")
                
                with detail_col2:
                    st.write("**✨ Key Strengths:**")
                    for highlight in cv.highlights:
                        st.write(f"• {highlight}")
                
                with detail_col3:
                    if cv.red_flags:
                        st.write("**🚨 Areas of Concern:**")
                        for flag in cv.red_flags:
                            st.write(f"• {flag}")
                    else:
                        st.write("**✅ No Major Concerns**")
                
                st.write("**📝 Analysis Summary:**")
                st.write(cv.rationale)
    
    # Summary table
    st.subheader("📋 Summary Table")
    
    results_data = []
    for i, cv in enumerate(sorted_cvs, 1):
        results_data.append({
            "Rank": i,
            "Name": cv.candidate_name or "N/A",
            "Score": f"{cv.score:.1f}",
            "Phone": cv.phone_number or "N/A",
            "Email": cv.email or "N/A",
            "Recommendation": cv.recommendation,
            "Summary": cv.rationale[:100] + "..." if len(cv.rationale) > 100 else cv.rationale
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

def export_results():
    """Export results to CSV"""
    all_cvs = sorted(st.session_state.cvs.values(), key=lambda x: x.score if x.score else 0, reverse=True)
    
    export_data = []
    for i, cv in enumerate(all_cvs, 1):
        export_data.append({
            "Rank": i,
            "Candidate_Name": cv.candidate_name or "",
            "Phone_Number": cv.phone_number or "",
            "Email": cv.email or "",
            "Filename": cv.filename,
            "Score": cv.score,
            "Recommendation": cv.recommendation,
            "Key_Strengths": "; ".join(cv.highlights) if cv.highlights else "",
            "Areas_of_Concern": "; ".join(cv.red_flags) if cv.red_flags else "",
            "Analysis_Summary": cv.rationale,
            "Upload_Time": cv.upload_ts.isoformat()
        })
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name=f"cv_analysis_{st.session_state.current_job.title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

def reset_analysis():
    """Reset analysis to start fresh"""
    st.session_state.analysis_complete = False
    st.session_state.cvs.clear()
    st.session_state.current_job = None

if __name__ == "__main__":
    main()
