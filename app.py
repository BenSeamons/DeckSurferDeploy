#!/usr/bin/env python3
# app.py - Flask backend for DeckSurfer (Heroku-optimized)
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

# Import your existing modules
from Deck_Sorter import (
    anki_invoke, find_notes, notes_info, set_suspended, change_deck, add_tag,
    index_candidate_pool, combined_top_candidates, EmbeddingIndex,
    extract_note_text, safe_norm, fuzzy_score, _EMBEDDINGS_OK
)

# Import PDF processing
try:
    from pdf_to_los import extract_text_from_pdf, clean_and_split

    PDF_PROCESSING_OK = True
except ImportError:
    PDF_PROCESSING_OK = False
    print("Warning: PDF processing not available. Install PyPDF2 to enable.")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

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
        'port': PORT
    })


@app.route('/api/anki/connection', methods=['GET'])
def check_anki_connection():
    """Test connection to AnkiConnect"""
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
                    return jsonify({'error': 'PDF processing not available'}), 500

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
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No configuration provided'}), 400

    try:
        # Extract configuration
        config = {
            'learning_objectives': data.get('learning_objectives', []),
            'target_deck': data.get('target_deck', ''),
            'source_decks': data.get('source_decks', ['AnKing Step Deck', 'USUHS v2.2']),
            'custom_tag': data.get('custom_tag'),
            'matching_mode': data.get('matching_mode', 'smart'),
            'auto_threshold': data.get('auto_threshold'),
            'multi_select': data.get('multi_select', False),
            'max_per_lo': data.get('max_per_lo', 3),
            'diversity_mode': data.get('diversity_mode', 'none'),
            'extra_query': data.get('extra_query'),
            'alpha': data.get('alpha', 0.6)
        }

        if not config['learning_objectives']:
            return jsonify({'error': 'No learning objectives provided'}), 400

        if not config['target_deck']:
            return jsonify({'error': 'Target deck is required'}), 400

        # Check AnkiConnect connection
        try:
            anki_invoke("version")
        except Exception as e:
            return jsonify({'error': f'Cannot connect to AnkiConnect: {str(e)}'}), 503

        # Build candidate pool
        logger.info(f"Building candidate pool from decks: {config['source_decks']}")
        pool = index_candidate_pool(
            decks=config['source_decks'],
            extra_query=config['extra_query']
        )

        if not pool:
            return jsonify({'error': 'No cards found in specified decks'}), 400

        logger.info(f"Found {len(pool)} candidate cards")

        # Initialize embedding index if needed
        emb_index = None
        if config['matching_mode'] == 'smart' and _EMBEDDINGS_OK:
            logger.info("Building embedding index...")
            emb_index = EmbeddingIndex()
            emb_index.fit(pool)

        # Process each learning objective
        results = []

        for i, lo in enumerate(config['learning_objectives']):
            logger.info(f"Processing LO {i + 1}/{len(config['learning_objectives'])}: {lo[:50]}...")

            # Get top candidates
            k_final = 10 if config['multi_select'] else 3
            candidates = combined_top_candidates(
                lo=lo,
                cards=pool,
                emb_index=emb_index,
                k_from_emb=80,
                k_final=k_final,
                alpha=config['alpha']
            )

            if not candidates:
                results.append({
                    'learning_objective': lo,
                    'matches': [],
                    'auto_selected': []
                })
                continue

            # Process candidates
            matches = []
            auto_selected = []

            for card, combo_score, fuzzy_score, emb_score in candidates:
                match_data = {
                    'note_id': card['noteId'],
                    'model_name': card.get('modelName', ''),
                    'tags': card.get('tags', []),
                    'fields': card.get('fields', {}),
                    'combined_score': round(combo_score, 3),
                    'fuzzy_score': round(fuzzy_score, 3),
                    'embedding_score': round(emb_score, 3),
                    'preview_text': extract_preview_text(card)
                }

                matches.append(match_data)

                # Auto-select if above threshold
                if (config['auto_threshold'] and
                        combo_score >= config['auto_threshold'] and
                        len(auto_selected) < config['max_per_lo']):
                    auto_selected.append(match_data)

            results.append({
                'learning_objective': lo,
                'matches': matches,
                'auto_selected': auto_selected
            })

        return jsonify({
            'results': results,
            'config': config,
            'stats': {
                'total_objectives': len(config['learning_objectives']),
                'total_candidates': len(pool),
                'embeddings_used': emb_index is not None
            }
        })

    except Exception as e:
        logger.error(f"Card matching failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/apply-changes', methods=['POST'])
def apply_changes_to_anki():
    """Apply the selected matches to Anki"""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        selected_matches = data.get('selected_matches', [])
        target_deck = data.get('target_deck')
        custom_tag = data.get('custom_tag')
        dry_run = data.get('dry_run', True)

        if not selected_matches:
            return jsonify({'error': 'No matches selected'}), 400

        if not target_deck:
            return jsonify({'error': 'Target deck is required'}), 400

        # Check AnkiConnect connection
        try:
            anki_invoke("version")
        except Exception as e:
            return jsonify({'error': f'Cannot connect to AnkiConnect: {str(e)}'}), 503

        if dry_run:
            return jsonify({
                'message': 'Dry run completed - no changes made to Anki',
                'would_modify': len(selected_matches),
                'target_deck': target_deck,
                'tag': custom_tag
            })

        # Apply changes to Anki
        note_ids = [match['note_id'] for match in selected_matches]

        # Unsuspend cards
        set_suspended(note_ids, suspended=False)

        # Move to target deck
        change_deck(note_ids, target_deck)

        # Add tag if specified
        if custom_tag:
            add_tag(note_ids, custom_tag)

        return jsonify({
            'message': f'Successfully processed {len(note_ids)} cards',
            'modified_notes': len(note_ids),
            'target_deck': target_deck,
            'tag_added': custom_tag
        })

    except Exception as e:
        logger.error(f"Failed to apply changes: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-results', methods=['POST'])
def export_results():
    """Export processing results as CSV"""
    data = request.get_json()

    try:
        results = data.get('results', [])
        selected_matches = data.get('selected_matches', [])

        # Create DataFrame for export
        export_data = []

        for result in results:
            lo = result['learning_objective']

            if result['matches']:
                for match in result['matches']:
                    is_selected = any(
                        sel['note_id'] == match['note_id'] and sel['learning_objective'] == lo
                        for sel in selected_matches
                    )

                    export_data.append({
                        'Learning_Objective': lo,
                        'Note_ID': match['note_id'],
                        'Model_Name': match['model_name'],
                        'Combined_Score': match['combined_score'],
                        'Fuzzy_Score': match['fuzzy_score'],
                        'Embedding_Score': match['embedding_score'],
                        'Preview_Text': match['preview_text'][:100] + '...' if len(match['preview_text']) > 100 else
                        match['preview_text'],
                        'Selected': 'Yes' if is_selected else 'No',
                        'Tags': ', '.join(match.get('tags', []))
                    })
            else:
                export_data.append({
                    'Learning_Objective': lo,
                    'Note_ID': '',
                    'Model_Name': '',
                    'Combined_Score': '',
                    'Fuzzy_Score': '',
                    'Embedding_Score': '',
                    'Preview_Text': 'No matches found',
                    'Selected': 'No',
                    'Tags': ''
                })

        df = pd.DataFrame(export_data)

        # Save to temporary file
        export_filename = f"decksurfer_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
        df.to_csv(export_path, index=False)

        return jsonify({
            'message': 'Results exported successfully',
            'filename': export_filename,
            'rows': len(export_data)
        })

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== HELPER FUNCTIONS ====================

def extract_preview_text(card: Dict[str, Any]) -> str:
    """Extract a readable preview from card fields"""
    fields = card.get('fields', {})

    # Priority order for fields to show
    field_priority = ['Front', 'Text', 'Back', 'Question', 'Answer']

    for field_name in field_priority:
        if field_name in fields and fields[field_name]:
            text = fields[field_name].strip()
            # Clean HTML tags if present
            import re
            text = re.sub(r'<[^>]+>', '', text)
            return text[:200]  # Truncate for preview

    # Fallback: use any available field
    for field_name, content in fields.items():
        if content and content.strip():
            import re
            text = re.sub(r'<[^>]+>', '', content.strip())
            return text[:200]

    return "No preview available"


# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


# ==================== STARTUP ====================

if __name__ == '__main__':
    print("🏄‍♂️ Starting DeckSurfer API Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Embeddings available: {_EMBEDDINGS_OK}")
    print(f"PDF processing available: {PDF_PROCESSING_OK}")
    print(f"Port: {PORT}")

    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Use Heroku's PORT or default to 5000
    app.run(host='0.0.0.0', port=PORT, debug=False)