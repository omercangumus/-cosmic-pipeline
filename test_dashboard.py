"""Simple test dashboard to verify Streamlit works."""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Test Dashboard", layout="wide")

st.title("🛰️ Cosmic Pipeline - Test Dashboard")
st.markdown("**TUA Astro Hackathon 2026**")

st.success("✅ Dashboard is working!")

# Test data generation
if st.button("Generate Test Data"):
    df = pd.DataFrame({
        'x': np.arange(100),
        'y': np.random.randn(100).cumsum()
    })
    
    st.line_chart(df.set_index('x'))
    st.success(f"Generated {len(df)} samples")

st.markdown("---")
st.markdown("If you see this, Streamlit is working correctly!")
