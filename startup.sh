#!/bin/bash
echo "Starting Streamlit app..."
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
