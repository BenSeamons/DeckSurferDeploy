#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# app.py - Flask backend for DeckSurfer (Unicode-safe with emojis)
import os
import tempfile
import uuid
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Railway-friendly
PORT = int(os.environ.get('PORT', 5000))
RAILWAY_ENVIRONMENT = os.environ.get('RAILWAY_ENVIRONMENT_NAME', 'development')
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'decksurfer_uploads')
SESSION_FOLDER = os.path.join(tempfile.gettempdir(), 'decksurfer_sessions')
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Try to import dependencies
try:
    import PyPDF2

    PDF_AVAILABLE = True
    print("✅ PDF processing available")
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PDF processing not available")

try:
    from rapidfuzz import fuzz

    FUZZY_AVAILABLE = True
    print("✅ Fuzzy matching available")
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ Fuzzy matching not available")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    EMBEDDINGS_AVAILABLE = True
    print("✅ AI embeddings available")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ AI embeddings not available")

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_session(session_id: str, data: dict):
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    with open(session_path, 'w') as f:
        json.dump(data, f)


def load_session(session_id: str) -> Optional[dict]:
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


# Serve the HTML file separately to avoid Unicode issues in Python strings
@app.route('/')
def index():
    """Serve the main HTML interface from a separate file"""
    return send_from_directory('.', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'environment': 'production' if RAILWAY_ENVIRONMENT == 'production' else 'development',
            'railway_environment': RAILWAY_ENVIRONMENT,
            'message': 'DeckSurfer backend is running! 🏄‍♂️',
            'features': {
                'text_processing': True,
                'csv_processing': True,
                'pdf_processing': PDF_AVAILABLE,
                'anki_integration': False,
                'ai_embeddings': EMBEDDINGS_AVAILABLE,
                'fuzzy_matching': FUZZY_AVAILABLE
            },
            'emojis_work': '✅ Emojis are working! 🎉'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# PDF processing (keeping the same functions)
def extract_text_from_pdf(pdf_path: str) -> str:
    if not PDF_AVAILABLE:
        raise ImportError("PyPDF2 not available")

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")


def clean_and_split_pdf_text(text: str) -> List[str]:
    import re
    text = re.sub(r'\s+', ' ', text)
    patterns = [
        r'(?:Learning\s+Objective|Objective|LO)[s]?[:\-\s]*([^\.]{20,200})',
        r'(?:Students?\s+will|By\s+the\s+end|Upon\s+completion)[^\.]{20,200}',
        r'(?:Understand|Describe|Explain|Identify|Analyze|Compare|Define)[^\.]{15,150}',
        r'(?:\d+\.|\*|\-|\•)\s*([^\.]{15,200})',
    ]

    objectives = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if 10 < len(cleaned) < 300 and cleaned not in objectives:
                objectives.append(cleaned)

    if len(objectives) < 3:
        candidates = re.split(r'(?:\n|\d+\.|•|\*|\-)\s*', text)
        for candidate in candidates:
            cleaned = candidate.strip()
            if (20 < len(cleaned) < 250 and
                    any(word in cleaned.lower() for word in
                        ['understand', 'describe', 'explain', 'identify', 'analyze', 'compare', 'define', 'discuss',
                         'evaluate', 'demonstrate']) and
                    cleaned not in objectives):
                objectives.append(cleaned)

    final_objectives = []
    for obj in objectives[:20]:
        obj = re.sub(r'^(?:Learning\s+Objective[s]?[:\-\s]*|LO[:\-\s]*|\d+\.\s*)', '', obj, flags=re.IGNORECASE)
        obj = obj.strip()
        if obj and len(obj) > 10:
            final_objectives.append(obj)

    return final_objectives if final_objectives else ['No clear learning objectives found in PDF']


# API endpoints for file upload and processing
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use CSV or PDF files only.'}), 400

    try:
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")

        file.save(file_path)

        session_data = {
            'file_path': file_path,
            'filename': filename,
            'file_type': filename.rsplit('.', 1)[1].lower(),
            'upload_time': pd.Timestamp.now().isoformat()
        }
        save_session(session_id, session_data)

        logger.info(f"File uploaded: {filename}, Session ID: {session_id}")

        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/process/extract-los', methods=['POST'])
def extract_learning_objectives():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        los = []

        if 'session_id' in data:
            session_id = data['session_id']
            session = load_session(session_id)
            if not session:
                return jsonify({
                    'error': f'Session not found: {session_id}. Please upload the file again.'
                }), 400

            file_path = session['file_path']
            file_type = session['file_type']

            if not os.path.exists(file_path):
                return jsonify({'error': f'File not found. Please upload again.'}), 400

            if file_type == 'csv':
                try:
                    df = pd.read_csv(file_path)
                    col = None
                    for candidate in ["Objective", "objective", "LO", "lo", "Objectives", "objectives"]:
                        if candidate in df.columns:
                            col = candidate
                            break

                    if not col:
                        return jsonify({
                            'error': 'CSV must contain a column named "Objective", "LO", or "Objectives"',
                            'available_columns': list(df.columns)
                        }), 400

                    los = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]

                except Exception as csv_error:
                    return jsonify({'error': f'CSV processing failed: {str(csv_error)}'}), 500

            elif file_type == 'pdf':
                if not PDF_AVAILABLE:
                    return jsonify({'error': 'PDF processing not available in this deployment'}), 500

                try:
                    text = extract_text_from_pdf(file_path)
                    los = clean_and_split_pdf_text(text)

                except Exception as pdf_error:
                    return jsonify({'error': f'PDF processing failed: {str(pdf_error)}'}), 500

            else:
                return jsonify({'error': f'Unsupported file type: {file_type}'}), 400

        elif 'text' in data:
            text_input = data['text'].strip()
            los = [line.strip() for line in text_input.split('\n') if line.strip()]

        else:
            return jsonify({'error': 'No valid input method provided'}), 400

        if not los:
            return jsonify({'error': 'No learning objectives could be extracted'}), 400

        if 'session_id' in data:
            session = load_session(data['session_id'])
            if session:
                session['learning_objectives'] = los
                save_session(data['session_id'], session)

        return jsonify({
            'learning_objectives': los,
            'count': len(los),
            'message': f'Successfully extracted {len(los)} learning objectives'
        })

    except Exception as e:
        logger.error(f"LO extraction failed: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("🏄‍♂️ Starting DeckSurfer API Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Session folder: {SESSION_FOLDER}")
    print(f"Port: {PORT}")
    print(f"PDF processing: {'✅' if PDF_AVAILABLE else '❌'}")
    print(f"AI embeddings: {'✅' if EMBEDDINGS_AVAILABLE else '❌'}")
    print(f"Fuzzy matching: {'✅' if FUZZY_AVAILABLE else '❌'}")

    app.run(host='0.0.0.0', port=PORT, debug=False)