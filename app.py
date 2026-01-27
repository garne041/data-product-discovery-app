import streamlit as st
import streamlit.components.v1 as components
import json
import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
from model_serving_utils import (
    query_endpoint, 
    query_endpoint_stream, 
    extract_content_from_stream,
    is_endpoint_supported,
    parse_rag_response
)

import time
import requests
from databricks.sdk.core import Config

cfg = Config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="FNMA Data Discovery",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ensure environment variable is set correctly
SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')
assert SERVING_ENDPOINT, \
    ("Unable to determine serving endpoint to use for chatbot app. If developing locally, "
     "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
     "deploying to a Databricks app, include a serving endpoint resource named "
     "'serving_endpoint' with CAN_QUERY permissions.")

# Get catalog and schema from environment variables for access requests
ACCESS_REQUEST_CATALOG = os.getenv('ACCESS_REQUEST_CATALOG', 'fnma_product_catalog')
ACCESS_REQUEST_SCHEMA = os.getenv('ACCESS_REQUEST_SCHEMA', 'default')

# Check if the endpoint is supported
endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

# Custom CSS for compact, modern styling
st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    
    /* Compact search header */
    .search-header {
        background: linear-gradient(to right, #ffffff 0%, #f8f9ff 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 2rem;
        border-left: 5px solid #00205b;
    }
    
    .logo-section {
        flex-shrink: 0;
        display: flex;
        align-items: center;
    }
    
    .title-section {
        flex: 1;
    }
    
    .title-section h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
        color: #00205b;
        letter-spacing: -0.5px;
    }
    
    .tagline-main {
        margin: 0 0 0.4rem 0;
        color: #2d2d2d;
        font-size: 1.15rem;
        line-height: 1.5;
        font-weight: 500;
    }
    
    .tagline-sub {
        margin: 0;
        color: #666;
        font-size: 0.85rem;
        line-height: 1.4;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Result card styling - compact with fixed structure */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
        border-top: 4px solid;
        display: flex;
        flex-direction: column;
        min-height: 620px;
        max-height: 620px;
    }
    
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .rank-1 { border-top-color: #FFD700; }
    .rank-2 { border-top-color: #C0C0C0; }
    .rank-3 { border-top-color: #CD7F32; }
    
    /* Compact card header - fixed height */
    .card-header {
        margin-bottom: 0.8rem;
        min-height: 85px;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.3rem;
        min-height: 26px;
    }
    
    .card-identifier {
        font-size: 0.7rem;
        color: #666;
        font-family: monospace;
        background: #f5f5f5;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        display: inline-block;
        margin-top: 0.3rem;
    }
    
    /* Compact badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    
    .badge-rank {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Card description - fixed height */
    .card-description {
        font-size: 0.85rem;
        color: #444;
        line-height: 1.4;
        margin: 0.8rem 0;
        min-height: 100px;
        max-height: 100px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Health status indicator */
    .health-status {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-size: 0.75rem;
        color: #666;
        margin-right: 0.5rem;
    }
    
    .health-light {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 4px rgba(0,0,0,0.2);
    }
    
    .health-green { background: #4caf50; box-shadow: 0 0 6px #4caf50; }
    .health-yellow { background: #ffc107; box-shadow: 0 0 6px #ffc107; }
    .health-red { background: #f44336; box-shadow: 0 0 6px #f44336; }
    
    /* Inline metadata in card */
    .card-inline-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        align-items: center;
        padding: 0.8rem 0;
        border-top: 1px solid #f0f0f0;
        border-bottom: 1px solid #f0f0f0;
        margin: 0.8rem 0;
        font-size: 0.75rem;
    }
    
    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.3rem;
        color: #666;
    }
    
    .meta-item strong {
        color: #1a1a1a;
    }
    
    /* Business value in card - compact */
    .card-business-value {
        font-size: 0.8rem;
        color: #444;
        line-height: 1.3;
        margin: 0.8rem 0;
        padding: 0.6rem;
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-radius: 6px;
        border-left: 3px solid #667eea;
        min-height: 65px;
        max-height: 65px;
        overflow: hidden;
    }
    
    .business-value-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Unity Catalog link in card */
    .unity-link {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.4rem 0.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        text-decoration: none;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .unity-link:hover {
        transform: scale(1.02);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        color: white !important;
        text-decoration: none;
    }
    
    /* Compact expander sections */
    .compact-section {
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid #f0f0f0;
    }
    
    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    /* Compact field list */
    .field-compact {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.4rem;
        border-left: 2px solid #667eea;
    }
    
    .field-compact-name {
        font-weight: 600;
        color: #1a1a1a;
        font-size: 0.75rem;
    }
    
    .field-compact-type {
        color: #667eea;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    
    .field-compact-desc {
        color: #666;
        font-size: 0.72rem;
        margin-top: 0.2rem;
    }
    
    /* Query understanding compact */
    .query-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .query-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .query-text {
        font-size: 0.95rem;
        color: #1a1a1a;
        font-weight: 500;
    }
    
    /* Recommended action compact */
    .recommended-compact {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 3px solid #4caf50;
    }
    
    .recommended-compact h3 {
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        color: #1a1a1a;
    }
    
    .recommended-compact p {
        font-size: 0.85rem;
        color: #444;
        margin: 0;
        line-height: 1.4;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Input field styling - compact */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
        border: 2px solid #e0e0e0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling - compact */
    .stButton > button {
        border-radius: 25px;
        padding: 0.6rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Streamlit expander styling - uniform sizing with better contrast */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        padding: 0.8rem !important;
        background: white !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        margin-top: 0 !important;
    }
    
    .streamlit-expanderContent {
        font-size: 0.8rem !important;
        padding: 1rem !important;
        background: white !important;
        border-radius: 0 0 10px 10px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        margin-top: -5px !important;
    }
    
    /* Ensure expanders have consistent heights - 0.5rem gap for tight spacing */
    [data-testid="stExpander"] {
        margin-top: 0.5rem !important;
        margin-bottom: 0 !important;
    }
    
    /* Field container styling */
    .field-box {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
    }
    
    .field-box:hover {
        background: #f0f2f5;
        border-color: #667eea;
        box-shadow: 0 2px 6px rgba(102, 126, 234, 0.1);
    }
    
    .field-name-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    
    .field-name-text {
        font-weight: 600;
        color: #1a1a1a;
        font-size: 0.85rem;
    }
    
    .field-type-badge {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: monospace;
    }
    
    .field-description {
        color: #555;
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    /* Ensure columns are perfectly equal with proper spacing */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
        padding: 0 0.5rem !important;
    }
    
    [data-testid="column"]:first-child {
        padding-left: 0 !important;
    }
    
    [data-testid="column"]:last-child {
        padding-right: 0 !important;
    }
    
    /* Remove default streamlit spacing that causes misalignment */
    .element-container {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Reduce spacing between HTML components and other elements */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0rem !important;
    }
    
    /* Specific targeting for HTML component containers */
    iframe {
        margin-bottom: 0 !important;
        display: block !important;
    }
    
    /* Loading and error styling */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    .stAlert {
        border-radius: 10px;
        font-size: 0.9rem;
    }
    
    /* Progress message styling */
    div[data-testid="stMarkdownContainer"] > div[data-testid="stMarkdown"] p {
        line-height: 1.5;
    }
    
    /* Info box styling for progress updates */
    .stAlert[data-baseweb="notification"] {
        padding: 0.75rem 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    </style>
""", unsafe_allow_html=True)

def render_search_header():
    """Render compact search header with FNMA logo"""
    st.markdown("""
        <div class="search-header">
            <div class="logo-section">
                <img src="https://www.smarthomeamerica.org/assets/uploads/fm_logo_4cp_nvy_c_r.png" 
                     alt="Fannie Mae Logo" 
                     style="height: 80px; width: auto;">
            </div>
            <div class="title-section">
                <h1>Data Discovery</h1>
                <p class="tagline-main">Simply ask what you need—our AI agent finds it</p>
                <p class="tagline-sub">Powered by Databricks agentic RAG • Instant results • Natural language search</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def get_health_status(last_updated):
    """Determine health status based on freshness"""
    if "continuous" in last_updated.lower() or "streaming" in last_updated.lower():
        return "green", "Live"
    elif "hour" in last_updated.lower() or "day" in last_updated.lower():
        return "yellow", "Recent"
    else:
        return "green", "Active"  # Default to green for demo
    

def convert_to_pandas_dataframe(table_names):
    return pd.DataFrame(eval(table_names))


def request_access_api(data_product_name):
    """Python function to handle access request"""
    try:
        workspace_url = cfg.host
        token = st.context.headers.get('X-Forwarded-Access-Token')

        # Build full_name from environment variables
        full_name = f"{ACCESS_REQUEST_CATALOG}.{ACCESS_REQUEST_SCHEMA}"

        payload = {
                "comment": "Requesting USE_SCHEMA permission",
                "securable": {
                    "full_name": full_name,
                    "type": "SCHEMA"
                },
                "privileges": "USE_SCHEMA,SELECT"
        }

        resp = requests.post(
            f"{workspace_url}/api/2.0/rfa/request",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        
        # Show custom toast message
        show_custom_toast(f"🚀 Access request sent successfully for {data_product_name}!", "success")
        
        return {"success": True, "message": f"Access request sent for {data_product_name}", "result": result}
        
    except Exception as e:
        logger.error(f"Failed to send access request: {e}")
        # Show error toast
        show_custom_toast(f"❌ Failed to send access request: {str(e)}", "error")
        return {"success": False, "message": f"Failed to send access request: {str(e)}"}


def show_custom_toast(message, toast_type="success"):
    """Display a custom toast message that disappears after 2 seconds"""
    toast_color = "#4caf50" if toast_type == "success" else "#f44336"
    
    toast_html = f"""
    <div id="custom-toast" style="
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        color: #1a1a1a;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        z-index: 999999;
        min-width: 400px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        border-left: 5px solid {toast_color};
        animation: slideIn 0.3s ease-out;
    ">
        {message}
    </div>
    <script>
        setTimeout(function() {{
            var toast = document.getElementById('custom-toast');
            if (toast) {{
                toast.style.animation = 'slideOut 0.3s ease-out';
                setTimeout(function() {{
                    toast.remove();
                }}, 300);
            }}
        }}, 2000);
    </script>
    <style>
        @keyframes slideIn {{
            from {{ 
                opacity: 0; 
                transform: translate(-50%, -50%) scale(0.8);
            }}
            to {{ 
                opacity: 1; 
                transform: translate(-50%, -50%) scale(1);
            }}
        }}
        @keyframes slideOut {{
            from {{ 
                opacity: 1; 
                transform: translate(-50%, -50%) scale(1);
            }}
            to {{ 
                opacity: 0; 
                transform: translate(-50%, -50%) scale(0.8);
            }}
        }}
    </style>
    """
    
    # Create a placeholder for the toast
    toast_placeholder = st.empty()
    toast_placeholder.markdown(toast_html, unsafe_allow_html=True)
    
    # Clear the placeholder after 2.5 seconds (2s display + 0.3s animation + buffer)
    time.sleep(2.5)
    toast_placeholder.empty()


def render_result_card_compact(result, container):
    """Render a compact result card with embedded DataFrame and request access button"""
    
    with container:
        rank = result['rank']
        rank_colors = {1: "🥇", 2: "🥈", 3: "🥉"}
        description = result['description']
        health_text = "Active"
        border_colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
        border_color = border_colors.get(rank, "#667eea")

        data_product_name = result['data_product_name']
        full_identifier = result['full_identifier']
        table_names = result['table_names']
        df = convert_to_pandas_dataframe(table_names)
        df_html = df.to_html(index=False, escape=False)
        
        table_html = f"""
            <div style="
                max-height:220px;
                overflow:auto;
                border:1px solid #ddd;
                border-radius:6px;
                padding:4px;
                margin-top:8px;
                background:white;
            ">
                {df_html}
            </div>
        """

        # HTML for result card (without JavaScript button)
        card_html = f'''
        <html>
        <head>
        <style>
            * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
            body {{ margin: 0; padding: 0; }}
            .result-card {{
                background: white;
                border-radius: 12px;
                padding: 1.25rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                border-top: 4px solid {border_color};
                height: 580px;
                display: flex;
                flex-direction: column;
                overflow-y: auto;
            }}
            .badge {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-right: 0.3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
            .card-title {{ font-size: 1.1rem; font-weight: 700; color: #1a1a1a; margin: 0.5rem 0 0.3rem 0; }}
            .card-identifier {{ font-size: 0.7rem; color: #666; font-family: monospace; background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 3px; display: inline-block; margin-top: 0.3rem; }}
            .card-description {{ font-size: 0.85rem; color: #444; line-height: 1.4; margin: 0.8rem 0; }}
            .card-inline-meta {{ 
                display: flex; 
                flex-wrap: wrap; 
                gap: 0.8rem; 
                align-items: center; 
                padding: 0.8rem 0; 
                border-top: 1px solid #f0f0f0; 
                border-bottom: 1px solid #f0f0f0; 
                margin: 0.8rem 0; 
                font-size: 0.75rem; 
            }}
            .health-status {{ display: inline-flex; align-items: center; gap: 0.3rem; color: #666; }}
            .health-light {{ width: 8px; height: 8px; border-radius: 50%; background: #4caf50; box-shadow: 0 0 6px #4caf50; }}
            table {{ border-collapse: collapse; width: 100%; font-size: 0.7rem; }}
            th, td {{ border: 1px solid #ddd; padding: 4px; text-align: center; }}
            th {{ background-color: #667eea; color: white; }}
            td:nth-child(2) {{ text-align: left; }}
        </style>
        </head>
        <body>
        <div class="result-card">
            <div>
                <span class="badge">{rank_colors.get(rank, "📊")} #{rank}</span>
                <div class="card-title">{data_product_name}</div>
                <div class="card-identifier">{full_identifier}</div>
            </div>
            <div class="card-description">{description}</div>
            <div class="card-inline-meta">
                <div class="health-status">
                    <span class="health-light"></span>
                    <span>{health_text}</span>
                </div>
            </div>
            {table_html}
        </div>
        </body>
        </html>
        '''

        # Render card
        components.html(card_html, height=580, scrolling=False)
        
        # Add Streamlit button below the card
        button_key = f"access_btn_{rank}_{data_product_name.replace(' ', '_')}"
        button_disabled = data_product_name in st.session_state.access_requested
        
        if st.button(
            "Request Sent ✓" if button_disabled else "Request Access", 
            key=button_key, 
            use_container_width=True,
            disabled=button_disabled
        ):
            st.session_state.access_requested.add(data_product_name)
            with st.spinner("Sending request..."):
                request_access_api(data_product_name)
            st.rerun()


def query_rag_endpoint(query, max_tokens=2000, progress_placeholder=None):
    """Query the RAG endpoint and parse the response with rotating progress updates"""
    
    try:
        messages = [{"role": "user", "content": query}]
        
        if progress_placeholder:
            progress_placeholder.info("🔧 Retrieving data product metadata")
        
        response = query_endpoint(
            endpoint_name=SERVING_ENDPOINT,
            messages=messages,
            max_tokens=max_tokens
        )
        
        logger.info(f"Raw response: {response}")
        
        if progress_placeholder:
            progress_placeholder.info("📊 **Processing results...** Extracting relevant data products")
            
            progress_placeholder.info("🎯 **Ranking matches...** Identifying best data products for your query")
        
        # Parse the RAG response to extract structured JSON
        parsed_data = parse_rag_response(response)
        
        if parsed_data:
            logger.info("Successfully parsed RAG response")
            if progress_placeholder:
                progress_placeholder.success("✨ **Results ready!** Found {} relevant data products".format(len(parsed_data.get('results', []))))
            return parsed_data
        else:
            logger.warning("Failed to parse RAG response, returning None")
            return None
            
    except Exception as e:
        logger.error(f"Error querying RAG endpoint: {e}")
        raise


def get_chatbot_response(user_input, chat_history):
    """Placeholder function for chatbot response - implement your logic here"""
    # This is a placeholder - implement your actual chatbot logic
    return f"Echo: {user_input}"


def main():
    # Initialize session state
    if 'searched' not in st.session_state:
        st.session_state.searched = False
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'results_data' not in st.session_state:
        st.session_state.results_data = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'access_requested' not in st.session_state:
        st.session_state.access_requested = set()
    
    # Render compact header
    render_search_header()
    
    # Check if endpoint is supported
    if not endpoint_supported:
        st.warning("⚠️ Could not validate endpoint. Proceeding anyway...")
    
    # Compact search input - single row
    tab1, tab2 = st.tabs(["📊 Data Discovery", "💬 Chatbot"])

    with tab1:
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Search",
                placeholder="e.g., Find data products about loans, customers, risk assessments, or credit analysis...",
                label_visibility="collapsed",
                key="search_input"
            )
        with col2:
            search_clicked = st.button("🔍 Search", use_container_width=True, key="search_button")
        
        if search_clicked and query:
            st.session_state.query = query
            
            # Create a single placeholder that gets replaced with each new message
            progress_placeholder = st.empty()
            
            # Query the RAG endpoint with rotating progress updates
            try:
                progress_placeholder.info("🚀 **Starting search...** Preparing to query data catalog")
                
                results_data = query_rag_endpoint(query, progress_placeholder=progress_placeholder)
                
                if results_data:
                    progress_placeholder.empty()  # Clear the final message
                    st.session_state.results_data = results_data
                    st.session_state.searched = True
                else:
                    progress_placeholder.empty()
                    st.error("❌ Could not parse the response from the RAG endpoint. Please try again.")
                    st.session_state.searched = False
                    
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Error querying endpoint: {str(e)}")
                st.info("💡 **Troubleshooting tips:**\n"
                    "- Verify your endpoint is running and accessible\n"
                    "- Check that you have the correct permissions\n"
                    "- Review the endpoint configuration")
                st.session_state.searched = False
        
        # Display results if searched
        if st.session_state.searched and st.session_state.results_data:
            data = st.session_state.results_data
            
            # Compact query understanding box
            if 'query_understanding' in data:
                st.markdown(f"""
                    <div class="query-box">
                        <div class="query-label">Query Understanding</div>
                        <div class="query-text">{data['query_understanding']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Results in 3 equal columns
            if 'results' in data and len(data['results']) > 0:
                # Determine number of columns based on results count
                num_results = min(len(data['results']), 3)
                cols = st.columns(num_results)
                
                for idx, result in enumerate(data['results'][:3]):
                    render_result_card_compact(result, cols[idx])
                
                # Compact recommended action
                if 'recommended_action' in data:
                    st.markdown(f"""
                        <div class="recommended-compact">
                            <h3>💡 Recommended Action</h3>
                            <p>{data['recommended_action']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No results found. Try refining your search query.")

    with tab2:
        st.header("💬 Interactive Chatbot")
        st.markdown("Chat with an AI assistant powered by Databricks OpenAI integration")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(user_input, st.session_state.chat_history)
                    st.write(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear chat history button in the main area
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
