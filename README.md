# DeckSurfer Deployment Guide 🚀

## Option 1: Quick Local Development Setup

### Prerequisites
- Python 3.9+ installed
- Anki with AnkiConnect add-on (Code: 2055492159)

### Steps
1. **Clone/Download the project files:**
   - `app.py` (Flask backend)
   - `Deck_Sorter.py` (your existing script)
   - `pdf_to_los.py` (your existing script) 
   - `requirements.txt`
   - `index.html` (frontend interface)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   - Open browser to `http://localhost:5000`
   - Make sure Anki is running with AnkiConnect enabled

## Option 2: Docker Deployment (Recommended)

### Prerequisites
- Docker and Docker Compose installed
- Domain name (optional, for public deployment)

### Steps
1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **Access the application:**
   - Local: `http://localhost:5000`
   - Production: `http://your-domain.com`

## Option 3: Cloud Deployment

### Heroku Deployment
1. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```

2. **Add Procfile:**
   ```
   web: python app.py
   ```

3. **Deploy:**
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

### Railway/Render Deployment
1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`
4. Deploy!

### DigitalOcean/AWS/GCP
1. Create a VM instance
2. Install Docker
3. Use Docker Compose setup above
4. Configure domain/SSL with Nginx

## Configuration

### Environment Variables
Set these in production:
```bash
FLASK_ENV=production
FLASK_DEBUG=False
MAX_FILE_SIZE=100MB  # Optional
UPLOAD_FOLDER=/app/uploads  # Optional
```

### SSL/HTTPS Setup
For production, configure SSL certificates:
1. Use Let's Encrypt with Certbot
2. Update nginx.conf with SSL configuration
3. Redirect HTTP to HTTPS

## File Structure
```
decksurfer/
├── app.py                 # Flask backend
├── Deck_Sorter.py         # Your existing matching logic
├── pdf_to_los.py          # PDF processing
├── requirements.txt       # Python dependencies
├── index.html            # Frontend interface
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-container setup
└── nginx.conf           # Reverse proxy config (optional)
```

## API Endpoints

The backend provides these REST API endpoints:

### Health & Connection
- `GET /api/health` - Backend health check
- `GET /api/anki/connection` - Test AnkiConnect
- `GET /api/anki/decks` - List available Anki decks

### File Processing  
- `POST /api/upload` - Upload CSV/PDF files
- `POST /api/process/extract-los` - Extract learning objectives

### Matching & Results
- `POST /api/process/match-cards` - Main matching algorithm
- `POST /api/apply-changes` - Apply selections to Anki
- `POST /api/export-results` - Export results as CSV/JSON

## Security Considerations

### For Local/Development Use
- App runs on localhost only
- No authentication needed (single user)
- AnkiConnect provides access control

### For Multi-User Deployment
- Add user authentication (Flask-Login)
- Implement rate limiting
- Secure file upload validation
- Use HTTPS in production
- Consider user data isolation

## Troubleshooting

### Common Issues
1. **AnkiConnect not connecting:**
   - Ensure Anki is running
   - Check AnkiConnect add-on is enabled
   - Verify no firewall blocking port 8765

2. **File upload fails:**
   - Check file size limits
   - Verify upload permissions
   - Ensure supported file formats (.csv, .pdf)

3. **Matching takes too long:**
   - Reduce deck size for testing
   - Disable embeddings if CPU-limited
   - Use auto-approval thresholds

4. **Memory issues:**
   - Limit candidate pool size
   - Process in smaller batches
   - Use lighter embedding models

### Performance Optimization
- **For large decks (>10k cards):**
  - Enable Redis caching
  - Use background task queues (Celery)
  - Implement pagination for results

- **For production:**
  - Use Gunicorn WSGI server
  - Configure nginx for static files
  - Enable gzip compression
  - Add CDN for assets

## Monitoring & Logging

### Basic Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Production Monitoring
- Use application monitoring (New Relic, DataDog)
- Set up error tracking (Sentry)
- Monitor disk usage for uploads
- Track API response times

## Backup & Maintenance

### Regular Maintenance
- Clean up old uploaded files
- Monitor disk usage
- Update dependencies regularly
- Backup user configurations

### Data Backup
- Export user results periodically  
- Backup application configurations
- Version control all code changes

## Scaling

### Single User → Multiple Users
1. Add user authentication
2. Implement user session management  
3. Isolate user data and uploads
4. Add user-specific configuration storage

### Performance Scaling
1. Use Redis for caching
2. Implement background job processing
3. Add load balancing for multiple instances
4. Use cloud storage for file uploads

---

## 🎯 Quick Start Commands

**Local Development:**
```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:5000
```

**Docker Deployment:**
```bash
docker-compose up -d
# Visit http://localhost:5000
```

**Production with SSL:**
```bash
# Configure domain in docker-compose.yml
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Your DeckSurfer app is now ready for medical students worldwide! 🏄‍♂️📚