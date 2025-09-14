#!/bin/bash
# Step 1: Run script to create/download model if not present
python create_model.py  

# Step 2: Start FastAPI with Uvicorn
uvicorn api.main:app --host 0.0.0.0 --port $PORT
