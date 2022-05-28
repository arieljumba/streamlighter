import streamlit as st
import pandas as pd
import numpy as np
from multiapp import MultiApp
from apps import timeseries

app = MultiApp()

st.set_page_config(layout='wide')
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
app.add_app("timeseries, timeseries.app)

# The main app
app.run()
