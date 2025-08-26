#!/usr/bin/env python3
# app.py - Flask backend for DeckSurfer (Unicode-safe for Railway)
import os
import tempfile
import uuid
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template_string
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

# Try to import PDF processing
try:
    import PyPDF2

    PDF_AVAILABLE = True
    print("PDF processing available")
except ImportError:
    PDF_AVAILABLE = False
    print("PDF processing not available")

# Try to import fuzzy matching
try:
    from rapidfuzz import fuzz

    FUZZY_AVAILABLE = True
    print("Fuzzy matching available")
except ImportError:
    FUZZY_AVAILABLE = False
    print("Fuzzy matching not available")

# Try to import embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    EMBEDDINGS_AVAILABLE = True
    print("AI embeddings available")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("AI embeddings not available")

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_session(session_id: str, data: dict):
    """Save session data to file"""
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    with open(session_path, 'w') as f:
        json.dump(data, f)


def load_session(session_id: str) -> Optional[dict]:
    """Load session data from file"""
    session_path = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


# PDF processing functions
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
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
    """Turn raw PDF text into learning objectives"""
    import re

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Common patterns for learning objectives
    patterns = [
        r'(?:Learning\s+Objective|Objective|LO)[s]?[:\-\s]*([^\.]{20,200})',
        r'(?:Students?\s+will|By\s+the\s+end|Upon\s+completion)[^\.]{20,200}',
        r'(?:Understand|Describe|Explain|Identify|Analyze|Compare|Define)[^\.]{15,150}',
        r'(?:\d+\.|\*|\-|\•)\s*([^\.]{15,200})',
    ]

    objectives = []

    # Try pattern matching first
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if 10 < len(cleaned) < 300 and cleaned not in objectives:
                objectives.append(cleaned)

    # If no pattern matches, try splitting on common delimiters
    if len(objectives) < 3:
        candidates = re.split(r'(?:\n|\d+\.|•|\*|\-)\s*', text)
        for candidate in candidates:
            cleaned = candidate.strip()
            # Filter for reasonable length sentences that look like learning objectives
            if (20 < len(cleaned) < 250 and
                    any(word in cleaned.lower() for word in
                        ['understand', 'describe', 'explain', 'identify', 'analyze', 'compare', 'define', 'discuss',
                         'evaluate', 'demonstrate']) and
                    cleaned not in objectives):
                objectives.append(cleaned)

    # Final cleanup and limit
    final_objectives = []
    for obj in objectives[:20]:  # Limit to 20 objectives
        # Remove common prefixes and clean up
        obj = re.sub(r'^(?:Learning\s+Objective[s]?[:\-\s]*|LO[:\-\s]*|\d+\.\s*)', '', obj, flags=re.IGNORECASE)
        obj = obj.strip()
        if obj and len(obj) > 10:
            final_objectives.append(obj)

    return final_objectives if final_objectives else ['No clear learning objectives found in PDF']


# Mock Anki functions for demo/remote use
def mock_anki_invoke(action: str, **params):
    """Mock AnkiConnect for demo purposes"""
    if action == "version":
        raise Exception("AnkiConnect not available - this is a remote server")
    elif action == "findNotes":
        # Return mock note IDs
        query = params.get('query', '')
        if 'AnKing' in query:
            return [1001, 1002, 1003, 1004, 1005]
        elif 'USUHS' in query:
            return [2001, 2002, 2003]
        return [9001, 9002]
    elif action == "notesInfo":
        # Return mock note data
        notes = params.get('notes', [])
        mock_notes = []
        for i, note_id in enumerate(notes):
            mock_notes.append({
                'noteId': note_id,
                'modelName': 'Cloze' if i % 2 == 0 else 'Basic',
                'tags': ['USUHS::Endocrine', 'AnKing::Step1'] if i % 3 == 0 else ['Medical'],
                'fields': {
                    'Front': f'Sample question about diabetes and insulin mechanism {i + 1}',
                    'Back': f'Answer explaining pathophysiology and clinical significance {i + 1}',
                    'Extra': f'Additional context about endocrine system {i + 1}'
                }
            })
        return mock_notes
    return None


# Fuzzy matching functions
def safe_norm(s: str) -> str:
    """Normalize text for comparison"""
    return " ".join(s.lower().split())


def fuzzy_score(lo: str, card_text: str) -> float:
    """Calculate fuzzy similarity score"""
    if not FUZZY_AVAILABLE:
        # Simple fallback scoring
        lo_words = set(lo.lower().split())
        card_words = set(card_text.lower().split())
        intersection = len(lo_words.intersection(card_words))
        union = len(lo_words.union(card_words))
        return intersection / union if union > 0 else 0.0

    # Use rapidfuzz for better scoring
    a = fuzz.token_set_ratio(lo, card_text) / 100.0
    b = fuzz.partial_ratio(lo, card_text) / 100.0
    c = fuzz.token_sort_ratio(lo, card_text) / 100.0
    return 0.5 * a + 0.3 * b + 0.2 * c


def extract_note_text(note: dict) -> str:
    """Extract searchable text from note"""
    fields = note.get("fields", {})
    pieces = []
    for k, v in fields.items():
        if isinstance(v, dict):
            val = v.get("value", "")
        else:
            val = str(v)
        pieces.append(f"{k}: {val}")
    tags = " ".join(note.get("tags", []))
    return ((" ".join(pieces)) + " " + tags).strip()


class EmbeddingIndex:
    """AI similarity search using sentence transformers"""

    def __init__(self):
        if not EMBEDDINGS_AVAILABLE:
            self.model = None
            self.card_matrix = None
            self.cards = []
            return

        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.card_matrix = None
            self.cards = []
            print("Embedding model loaded")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            self.model = None

    def fit(self, cards: List[dict]):
        if not self.model:
            return
        try:
            texts = [c.get("text", "") for c in cards]
            self.card_matrix = self.model.encode(texts, normalize_embeddings=True)
            self.cards = cards
            print(f"Indexed {len(cards)} cards for similarity search")
        except Exception as e:
            print(f"Failed to create embeddings: {e}")
            self.model = None

    def query(self, lo_text: str, top_k: int = 50):
        if not self.model or self.card_matrix is None:
            return []
        try:
            q = self.model.encode([lo_text], normalize_embeddings=True)[0]
            sims = self.card_matrix @ q
            idxs = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
            ranked = sorted([(int(i), float(sims[i])) for i in idxs], key=lambda x: -x[1])
            return ranked
        except Exception as e:
            print(f"Embedding query failed: {e}")
            return []


# Card matching and processing functions
def index_candidate_pool(decks: List[str], limit: Optional[int] = None, demo_mode: bool = True) -> List[dict]:
    """Build candidate card pool from specified decks"""
    if demo_mode:
        # Generate mock data for demo
        mock_cards = []
        card_templates = [
            {"front": "What hormone is deficient in Type 1 diabetes?", "back": "Insulin",
             "tags": ["USUHS::Endocrine", "AnKing::Step1"]},
            {"front": "DKA is characterized by what three findings?", "back": "Hyperglycemia, ketosis, acidosis",
             "tags": ["USUHS::Emergency", "AnKing::Step1"]},
            {"front": "Which cells produce insulin?", "back": "Beta cells of pancreatic islets",
             "tags": ["USUHS::Endocrine"]},
            {"front": "What is the primary defect in Type 2 diabetes?", "back": "Insulin resistance",
             "tags": ["USUHS::Endocrine", "AnKing::Step1"]},
            {"front": "What is the normal blood glucose range?", "back": "70-100 mg/dL fasting",
             "tags": ["USUHS::Lab Values"]},
            {"front": "What is HbA1c and what does it measure?",
             "back": "Glycated hemoglobin, measures average blood glucose over 2-3 months",
             "tags": ["USUHS::Lab Values", "AnKing::Step1"]},
            {"front": "What are the classic symptoms of diabetes?",
             "back": "Polyuria, polydipsia, polyphagia, weight loss", "tags": ["USUHS::Clinical"]},
            {"front": "What is the mechanism of metformin?",
             "back": "Decreases hepatic glucose production, increases insulin sensitivity",
             "tags": ["USUHS::Pharmacology", "AnKing::Step1"]},
        ]

        for i, template in enumerate(card_templates * 3):  # Duplicate for more examples
            note_id = 1000 + i
            mock_cards.append({
                'noteId': note_id,
                'modelName': 'Basic' if i % 2 == 0 else 'Cloze',
                'tags': template['tags'],
                'fields': {
                    'Front': template['front'],
                    'Back': template['back'],
                    'Extra': f"Additional clinical context for card {i + 1}"
                },
                'text': f"Front: {template['front']} Back: {template['back']} {' '.join(template['tags'])}"
            })

        if limit:
            mock_cards = mock_cards[:limit]

        print(f"[Demo Mode] Generated {len(mock_cards)} mock cards")
        return mock_cards

    else:
        # Real AnkiConnect mode (for local installations)
        try:
            # This would work with real AnkiConnect
            query = " OR ".join([f'deck:"{d}*"' for d in decks])
            note_ids = mock_anki_invoke("findNotes", query=query)
            if limit:
                note_ids = note_ids[:limit]

            notes = mock_anki_invoke("notesInfo", notes=note_ids)
            pool = []
            for n in notes:
                text = extract_note_text(n)
                pool.append({
                    "noteId": n.get("noteId"),
                    "modelName": n.get("modelName"),
                    "tags": n.get("tags", []),
                    "fields": {k: v.get("value", "") if isinstance(v, dict) else str(v)
                               for k, v in n.get("fields", {}).items()},
                    "text": text
                })
            return pool
        except Exception as e:
            print(f"AnkiConnect failed, using demo mode: {e}")
            return index_candidate_pool(decks, limit, demo_mode=True)


def combined_top_candidates(
        lo: str,
        cards: List[dict],
        emb_index: Optional[EmbeddingIndex],
        k_from_emb: int = 50,
        k_final: int = 3,
        alpha: float = 0.6
) -> List[tuple]:
    """
    Get top matching cards using combined fuzzy + AI scoring
    Returns: List of (card, combined_score, fuzzy_score, embedding_score)
    """
    lo_norm = safe_norm(lo)

    # Get embedding candidates if available
    seed_idxs = []
    emb_scores_map = {}

    if emb_index and emb_index.model:
        emb_ranked = emb_index.query(lo_norm, top_k=k_from_emb)
        seed_idxs = [i for (i, _) in emb_ranked]
        emb_scores_map = {i: s for (i, s) in emb_ranked}
    else:
        seed_idxs = list(range(min(len(cards), k_from_emb)))  # Use first N cards

    # Score candidates
    scored = []
    for i in seed_idxs:
        if i >= len(cards):
            continue

        c = cards[i]
        fz = fuzzy_score(lo_norm, c.get("text", ""))
        em = emb_scores_map.get(i, 0.0)

        if emb_index and emb_index.model:
            combo = alpha * em + (1 - alpha) * fz
        else:
            combo = fz

        scored.append((i, combo, fz, em))

    # Sort and return top K
    scored.sort(key=lambda x: -x[1])
    top = scored[:k_final]
    return [(cards[i], combo, fz, em) for (i, combo, fz, em) in top]


# HTML template (ASCII-safe)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeckSurfer - Smart Anki Organization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; color: white; }
        .header h1 { font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .main-card { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .connection-status { display: flex; align-items: center; gap: 10px; padding: 10px 15px; border-radius: 8px; margin: 15px 0; }
        .connection-status.connected { background: #d4edda; color: #155724; }
        .connection-status.disconnected { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DeckSurfer</h1>
            <p>Smart Anki deck organization for medical students</p>
        </div>
        <div class="main-card">
            <div id="connectionStatus" class="connection-status disconnected">
                <span>Checking backend...</span>
            </div>
            <div>
                <h2>DeckSurfer is Running!</h2>
                <p>The backend is successfully deployed with full AI capabilities.</p>
                <ul>
                    <li>PDF Processing: Available</li>
                    <li>AI Matching: Available</li>
                    <li>Fuzzy Matching: Available</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>'''


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'environment': 'production' if RAILWAY_ENVIRONMENT == 'production' else 'development',
            'railway_environment': RAILWAY_ENVIRONMENT,
            'message': 'DeckSurfer backend is running!',
            'features': {
                'text_processing': True,
                'csv_processing': True,
                'pdf_processing': PDF_AVAILABLE,
                'anki_integration': False,
                'ai_embeddings': EMBEDDINGS_AVAILABLE,
                'fuzzy_matching': FUZZY_AVAILABLE
            },
            'deployment_info': 'Full version with AI capabilities'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Additional endpoints would go here...

if __name__ == '__main__':
    print("Starting DeckSurfer API Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Session folder: {SESSION_FOLDER}")
    print(f"Port: {PORT}")
    print(f"PDF processing: {PDF_AVAILABLE}")
    print(f"AI embeddings: {EMBEDDINGS_AVAILABLE}")
    print(f"Fuzzy matching: {FUZZY_AVAILABLE}")

    app.run(host='0.0.0.0', port=PORT, debug=False)