#!/usr/bin/env python3
# app.py - Flask backend for DeckSurfer (Fixed for Heroku)
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import requests
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Heroku-friendly
PORT = int(os.environ.get('PORT', 5000))
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'decksurfer_uploads')
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global state for processing sessions
processing_sessions: Dict[str, Dict] = {}

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Inline HTML template (since templates folder might not be working)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeckSurfer - Smart Anki Organization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }

        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .main-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 15px 0;
        }

        .connection-status.connected {
            background: #d4edda;
            color: #155724;
        }

        .connection-status.disconnected {
            background: #f8d7da;
            color: #721c24;
        }

        .step {
            margin-bottom: 40px;
        }

        .step-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }

        .step-number {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }

        .step-title {
            font-size: 1.5rem;
            color: #333;
        }

        .form-group {
            margin-bottom: 25px;
        }

        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #555;
        }

        input, select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .file-drop-zone {
            border: 3px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .file-drop-zone:hover, .file-drop-zone.dragover {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .file-info {
            margin-top: 10px;
            font-size: 0.9rem;
            color: #666;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .hidden {
            display: none;
        }

        .los-preview {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            max-height: 200px;
            overflow-y: auto;
        }

        .status-indicator {
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-weight: 600;
        }

        .status-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .status-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }

        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .main-card {
                padding: 20px;
            }

            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏄‍♂️ DeckSurfer</h1>
            <p>Smart Anki deck organization for medical students</p>
        </div>

        <div class="main-card">
            <!-- Connection Status -->
            <div id="connectionStatus" class="connection-status disconnected">
                <span>🔍</span>
                <span id="connectionText">Checking backend connection...</span>
            </div>

            <!-- Step 1: Upload Learning Objectives -->
            <div class="step">
                <div class="step-header">
                    <div class="step-number">1</div>
                    <h2 class="step-title">Upload Learning Objectives</h2>
                </div>

                <div class="form-group">
                    <label>Choose your input method:</label>
                    <select id="inputMethod" onchange="toggleInputMethod()">
                        <option value="text">Manual Text Input</option>
                        <option value="csv">CSV File with Learning Objectives</option>
                        <option value="pdf">PDF Lecture (Auto-extract objectives)</option>
                    </select>
                </div>

                <!-- Manual Text Input -->
                <div id="textInput" class="form-group">
                    <label>Enter Learning Objectives (one per line):</label>
                    <textarea id="manualObjectives" rows="8" placeholder="Enter each learning objective on a new line...

Example:
Describe the pathophysiology of diabetes mellitus
Explain the mechanism of action of insulin
Identify clinical signs of diabetic ketoacidosis"></textarea>
                </div>

                <!-- CSV Upload -->
                <div id="csvInput" class="form-group hidden">
                    <label>Upload CSV File:</label>
                    <div class="file-drop-zone" onclick="document.getElementById('csvFile').click()">
                        <div>📄 Drop your CSV file here or click to browse</div>
                        <div class="file-info">Should contain a column named 'Objective', 'LO', or 'Objectives'</div>
                    </div>
                    <input type="file" id="csvFile" accept=".csv" style="display: none;" onchange="handleFileSelect(event, 'csv')">
                    <div id="csvFileInfo" class="file-info"></div>
                </div>

                <!-- PDF Upload -->
                <div id="pdfInput" class="form-group hidden">
                    <label>Upload PDF Lecture:</label>
                    <div class="file-drop-zone" onclick="document.getElementById('pdfFile').click()">
                        <div>📚 Drop your lecture PDF here or click to browse</div>
                        <div class="file-info">We'll automatically extract learning objectives from the content</div>
                    </div>
                    <input type="file" id="pdfFile" accept=".pdf" style="display: none;" onchange="handleFileSelect(event, 'pdf')">
                    <div id="pdfFileInfo" class="file-info"></div>
                </div>

                <!-- Preview Extracted LOs -->
                <div id="losPreview" class="los-preview hidden">
                    <h4>📋 Extracted Learning Objectives:</h4>
                    <ol id="losPreviewList"></ol>
                    <div class="file-info">
                        <span id="losCount">0</span> objectives found.
                    </div>
                </div>

                <button class="btn btn-primary" onclick="extractObjectives()" id="extractBtn">
                    📖 Extract Learning Objectives
                </button>
            </div>

            <!-- Status Messages -->
            <div id="statusIndicator" class="status-indicator hidden">
                <div id="statusText">Processing...</div>
            </div>
        </div>

        <!-- Info Panel -->
        <div class="main-card">
            <h3>📋 About DeckSurfer</h3>
            <p style="margin: 20px 0; line-height: 1.6;">
                This is a <strong>demo version</strong> of DeckSurfer deployed on Heroku. 
                It can extract and process learning objectives, but the full Anki integration 
                requires a local installation with AnkiConnect.
            </p>

            <h3>🚀 For Full Functionality:</h3>
            <ul style="margin: 20px 0; padding-left: 30px;">
                <li>Clone the repository from GitHub</li>
                <li>Install locally with Python</li>
                <li>Run Anki with AnkiConnect add-on</li>
                <li>Enjoy full card matching and organization!</li>
            </ul>
        </div>
    </div>

    <script>
        // Global state
        let extractedLOs = [];
        let sessionId = null;

        // Initialize app
        document.addEventListener('DOMContentLoaded', function() {
            toggleInputMethod();
            checkConnection();
        });

        // ==================== CONNECTION MANAGEMENT ====================
        async function checkConnection() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();

                const statusEl = document.getElementById('connectionStatus');
                const textEl = document.getElementById('connectionText');

                if (response.ok && data.status === 'healthy') {
                    statusEl.className = 'connection-status connected';
                    textEl.textContent = `✅ Backend connected successfully! (${data.environment || 'production'} mode)`;
                } else {
                    statusEl.className = 'connection-status disconnected';
                    textEl.textContent = `❌ Backend connection failed`;
                }
            } catch (error) {
                const statusEl = document.getElementById('connectionStatus');
                const textEl = document.getElementById('connectionText');
                statusEl.className = 'connection-status disconnected';
                textEl.textContent = '❌ Cannot connect to backend server';
                console.error('Connection error:', error);
            }
        }

        // ==================== UI MANAGEMENT ====================
        function toggleInputMethod() {
            const method = document.getElementById('inputMethod').value;
            document.getElementById('csvInput').classList.toggle('hidden', method !== 'csv');
            document.getElementById('pdfInput').classList.toggle('hidden', method !== 'pdf');
            document.getElementById('textInput').classList.toggle('hidden', method !== 'text');
        }

        async function handleFileSelect(event, type) {
            const file = event.target.files[0];
            if (!file) return;

            const infoElement = document.getElementById(type + 'FileInfo');
            infoElement.innerHTML = `✅ Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;

            // Upload file to server
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    sessionId = data.session_id;
                    infoElement.innerHTML += ` - Uploaded successfully`;
                } else {
                    throw new Error(data.error);
                }
            } catch (error) {
                infoElement.innerHTML = `❌ Upload failed: ${error.message}`;
            }
        }

        // ==================== LEARNING OBJECTIVES EXTRACTION ====================
        async function extractObjectives() {
            const method = document.getElementById('inputMethod').value;
            const extractBtn = document.getElementById('extractBtn');
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');

            extractBtn.disabled = true;
            extractBtn.textContent = '🔄 Extracting...';
            statusIndicator.classList.remove('hidden');
            statusIndicator.className = 'status-indicator';
            statusText.textContent = 'Processing learning objectives...';

            try {
                let requestData = {};

                if (method === 'text') {
                    const textInput = document.getElementById('manualObjectives').value.trim();
                    if (!textInput) {
                        throw new Error('Please enter some learning objectives');
                    }
                    requestData.text = textInput;
                } else if (method === 'csv' || method === 'pdf') {
                    if (!sessionId) {
                        throw new Error('Please upload a file first');
                    }
                    requestData.session_id = sessionId;
                }

                const response = await fetch('/api/process/extract-los', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });

                const data = await response.json();

                if (response.ok) {
                    extractedLOs = data.learning_objectives;
                    displayLOsPreview();
                    statusIndicator.className = 'status-indicator status-success';
                    statusText.textContent = `✅ Successfully extracted ${data.count} learning objectives!`;
                } else {
                    throw new Error(data.error);
                }
            } catch (error) {
                statusIndicator.className = 'status-indicator status-error';
                statusText.textContent = `❌ Error: ${error.message}`;
            } finally {
                extractBtn.disabled = false;
                extractBtn.textContent = '📖 Extract Learning Objectives';
            }
        }

        function displayLOsPreview() {
            const previewEl = document.getElementById('losPreview');
            const listEl = document.getElementById('losPreviewList');
            const countEl = document.getElementById('losCount');

            listEl.innerHTML = '';
            extractedLOs.forEach((lo, index) => {
                const li = document.createElement('li');
                li.textContent = lo;
                li.style.marginBottom = '5px';
                listEl.appendChild(li);
            });

            countEl.textContent = extractedLOs.length;
            previewEl.classList.remove('hidden');
        }
    </script>
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
    return jsonify({
        'status': 'healthy',
        'embeddings_available': False,
        'pdf_processing_available': False,
        'deck_sorter_available': False,
        'port': PORT,
        'environment': 'demo',
        'message': 'DeckSurfer demo backend is running!'
    })


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
                return jsonify({'error': 'PDF processing not available in demo version'}), 500

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
    print("🏄‍♂️ Starting DeckSurfer Demo Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Port: {PORT}")

    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Use Heroku's PORT or default to 5000
    app.run(host='0.0.0.0', port=PORT, debug=False)