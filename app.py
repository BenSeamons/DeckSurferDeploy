#!/usr/bin/env python3
# app.py - Flask backend for DeckSurfer (Heroku Production)
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import requests
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging

# Import your existing modules with error handling
try:
    from Deck_Sorter import (
        anki_invoke, find_notes, notes_info, set_suspended, change_deck, add_tag,
        index_candidate_pool, combined_top_candidates, EmbeddingIndex,
        extract_note_text, safe_norm, fuzzy_score, _EMBEDDINGS_OK
    )

    DECK_SORTER_OK = True
except ImportError as e:
    print(f"Warning: Deck_Sorter import failed: {e}")
    DECK_SORTER_OK = False
    _EMBEDDINGS_OK = False

# Import PDF processing with error handling
try:
    from pdf_to_los import extract_text_from_pdf, clean_and_split

    PDF_PROCESSING_OK = True
except ImportError as e:
    print(f"Warning: PDF processing not available: {e}")
    PDF_PROCESSING_OK = False

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Heroku-friendly
PORT = int(os.environ.get('PORT', 5000))
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', tempfile.mkdtemp())
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global state for processing sessions
processing_sessions: Dict[str, Dict] = {}


# Dummy implementations for when imports fail
def dummy_anki_invoke(action: str, **params):
    if action == "version":
        raise Exception("AnkiConnect not available - this is a demo deployment")
    return None


if not DECK_SORTER_OK:
    anki_invoke = dummy_anki_invoke


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'embeddings_available': _EMBEDDINGS_OK,
        'pdf_processing_available': PDF_PROCESSING_OK,
        'deck_sorter_available': DECK_SORTER_OK,
        'port': PORT,
        'environment': 'production' if not app.debug else 'development'
    })


@app.route('/api/anki/connection', methods=['GET'])
def check_anki_connection():
    """Test connection to AnkiConnect"""
    if not DECK_SORTER_OK:
        return jsonify({
            'connected': False,
            'error': 'This is a demo deployment. AnkiConnect integration requires local installation.'
        }), 503

    try:
        result = anki_invoke("version")
        return jsonify({
            'connected': True,
            'version': result
        })
    except Exception as e:
        logger.error(f"AnkiConnect connection failed: {e}")
        return jsonify({
            'connected': False,
            'error': str(e)
        }), 503


@app.route('/api/anki/decks', methods=['GET'])
def get_available_decks():
    """Get list of available Anki decks"""
    if not DECK_SORTER_OK:
        return jsonify({
            'error': 'This is a demo deployment. Full functionality requires local installation.'
        }), 500

    try:
        decks = anki_invoke("deckNames")
        return jsonify({'decks': decks})
    except Exception as e:
        logger.error(f"Failed to get decks: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads (CSV or PDF)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Generate unique filename
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)

        # Store file info in session
        processing_sessions[session_id] = {
            'file_path': file_path,
            'filename': filename,
            'file_type': filename.rsplit('.', 1)[1].lower(),
            'upload_time': pd.Timestamp.now().isoformat()
        }

        return jsonify({
            'session_id': session_id,
            'filename': filename,
            'message': 'File uploaded successfully'
        })

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/extract-los', methods=['POST'])
def extract_learning_objectives():
    """Extract learning objectives from uploaded file or text"""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        los = []

        if 'session_id' in data:
            # Process uploaded file
            session_id = data['session_id']
            if session_id not in processing_sessions:
                return jsonify({'error': 'Invalid session ID'}), 400

            session = processing_sessions[session_id]
            file_path = session['file_path']
            file_type = session['file_type']

            if file_type == 'csv':
                # Process CSV file
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

            elif file_type == 'pdf':
                # Process PDF file
                if not PDF_PROCESSING_OK:
                    return jsonify({'error': 'PDF processing not available in this deployment'}), 500

                text = extract_text_from_pdf(file_path)
                los = clean_and_split(text)

        elif 'text' in data:
            # Process manual text input
            text_input = data['text'].strip()
            los = [line.strip() for line in text_input.split('\n') if line.strip()]

        else:
            return jsonify({'error': 'No valid input method provided'}), 400

        if not los:
            return jsonify({'error': 'No learning objectives could be extracted'}), 400

        # Store LOs in session for later processing
        if 'session_id' in data:
            processing_sessions[data['session_id']]['learning_objectives'] = los

        return jsonify({
            'learning_objectives': los,
            'count': len(los)
        })

    except Exception as e:
        logger.error(f"LO extraction failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process/match-cards', methods=['POST'])
def match_cards_to_objectives():
    """Main processing endpoint - match cards to learning objectives"""
    if not DECK_SORTER_OK:
        return jsonify({
            'error': 'This is a demo deployment. Card matching requires local installation with AnkiConnect.'
        }), 500

    # [Rest of the matching logic would go here - same as before]
    return jsonify({
        'error': 'Card matching not available in demo deployment'
    }), 500


@app.route('/api/apply-changes', methods=['POST'])
def apply_changes_to_anki():
    """Apply the selected matches to Anki"""
    if not DECK_SORTER_OK:
        return jsonify({
            'error': 'This is a demo deployment. Anki modifications require local installation.'
        }), 500

    return jsonify({
        'error': 'Anki modifications not available in demo deployment'
    }), 500


@app.route('/api/export-results', methods=['POST'])
def export_results():
    """Export processing results as CSV"""
    try:
        # This can work in demo mode
        return jsonify({
            'message': 'Export functionality available in full version',
            'demo': True
        })
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


# ==================== STARTUP ====================

if __name__ == '__main__':
    print("🏄‍♂️ Starting DeckSurfer API Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Embeddings available: {_EMBEDDINGS_OK}")
    print(f"PDF processing available: {PDF_PROCESSING_OK}")
    print(f"Deck sorter available: {DECK_SORTER_OK}")
    print(f"Port: {PORT}")

    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Use Heroku's PORT or default to 5000
    app.run(host='0.0.0.0', port=PORT, debug=False)