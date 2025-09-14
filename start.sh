#!/bin/bash
# Create model folder and model file if not exists
python create_model.py

# Start backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
