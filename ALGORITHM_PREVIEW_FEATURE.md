# Algorithm Preview Feature

## Overview

The Algorithm Preview feature allows you to view, manage, and download the generated trading algorithms created during simulations. This feature provides a convenient UI to inspect the AI-generated Python code, copy snippets, download files, or clean up old algorithms.

## Features

- **📋 List View**: See all generated algorithm files with metadata (size, last modified)
- **👁️ Preview**: View algorithm code in a syntax-highlighted, read-only viewer
- **📄 Copy**: Copy algorithm code to clipboard with one click
- **💾 Download**: Download individual algorithm files
- **🗑️ Delete**: Remove individual files or bulk delete all algorithms
- **⚙️ Auto-Delete**: Optional toggle to automatically delete algorithms after previewing
- **🔄 Auto-Refresh**: Polls every 5 seconds to detect newly generated files

## How to Use

### Frontend (User Interface)

1. **Open Preview Modal**
   - Click the "📄 Preview Algorithms" button located below the ALGOCLASH title on the Dashboard

2. **Browse Algorithms**
   - The left panel shows all available algorithm files
   - Files are sorted by modification time (newest first)
   - Each entry shows: model name, file size, and timestamp

3. **Preview Code**
   - Click on any algorithm in the list to view its code
   - The right panel displays the full Python source code
   - Code is displayed in a monospace font with proper formatting

4. **Actions**
   - **Copy**: Click "📋 Copy" to copy code to clipboard
   - **Download**: Click "💾 Download" to save the file locally
   - **Delete One**: Click the 🗑️ icon next to a file to delete it
   - **Delete All**: Click "Delete All" button to remove all algorithm files

5. **Auto-Delete Mode**
   - Toggle "Auto-delete after preview" to enable automatic cleanup
   - When enabled, closing a preview will prompt to delete that file
   - Useful for one-time code inspection

### Backend API Endpoints

All endpoints are under `/api/algos` with CORS enabled.

#### `GET /api/algos`
List all generated algorithm files.

**Response**: JSON array
```json
[
  {
    "filename": "generated_algo_gpt-4.py",
    "modelName": "gpt-4",
    "sizeBytes": 2048,
    "createdAt": "2025-10-21T10:30:00",
    "modifiedAt": "2025-10-21T10:30:00"
  }
]
```

#### `GET /api/algos/:filename`
Get the raw text content of a specific algorithm file.

**Response**: Plain text (Python code)
```python
def execute_trade(ticker, cash_balance, shares_held):
    # AI-generated trading logic
    return "BUY"
```

#### `GET /api/algos/:filename/download`
Download an algorithm file as an attachment.

**Response**: File download

#### `DELETE /api/algos/:filename`
Delete a single algorithm file.

**Response**: JSON
```json
{
  "deleted": true,
  "filename": "generated_algo_gpt-4.py"
}
```

#### `DELETE /api/algos`
Delete all generated algorithm files (bulk operation).

**Response**: JSON
```json
{
  "deleted": 3,
  "message": "Deleted 3 algorithm file(s)"
}
```

## Security

The API implements multiple security layers:

1. **Filename Whitelist**: Only files matching `generated_algo_*.py` are accessible
2. **Path Sanitization**: All filenames are validated to prevent directory traversal
3. **Directory Restriction**: All operations are strictly limited to `backend/generate_algo/`
4. **No Execution**: Algorithms are only read/deleted, never executed via this API
5. **Input Validation**: Regex pattern matching and path resolution checks

### Blocked Attack Vectors
- Directory traversal: `../../../etc/passwd` ❌
- Path separators: `test/generated_algo.py` ❌
- Wrong extensions: `generated_algo_test.txt` ❌
- Null byte injection: `file.py\x00.txt` ❌

## Running Tests

### Backend Unit Tests

```bash
cd backend

# Activate virtual environment
source venv/bin/activate

# Run tests
python -m unittest tests.test_algos_api -v
```

**Test Coverage:**
- ✅ Security: filename validation, path sanitization
- ✅ List algorithms: empty, with files, metadata
- ✅ Get algorithm: success, not found, invalid filename
- ✅ Delete algorithm: success, not found
- ✅ Delete all: bulk operation
- ✅ Download: file attachment
- ✅ Edge cases: null bytes, empty inputs, directory traversal

### Manual Testing

1. **Start Backend**
   ```bash
   cd backend
   source venv/bin/activate
   python app.py
   # Backend runs on http://localhost:5000
   ```

2. **Start Frontend**
   ```bash
   cd frontend
   npm install  # First time only
   npm start
   # Frontend runs on http://localhost:3000
   ```

3. **Test Workflow**
   - Run a simulation to generate algorithms
   - Click "Preview Algorithms" button
   - Verify files appear in the modal
   - Click on a file to preview
   - Test copy, download, and delete actions
   - Enable auto-delete and test the workflow

## File Structure

### Backend
```
backend/
├── api/
│   ├── __init__.py          # Package marker
│   └── algos.py             # Algorithm API blueprint (NEW)
├── app.py                   # Flask app (MODIFIED: registers blueprint)
├── generate_algo/           # Generated algorithms directory
│   └── generated_algo_*.py  # AI-generated trading algorithms
└── tests/
    ├── __init__.py          # Package marker
    └── test_algos_api.py    # API unit tests (NEW)
```

### Frontend
```
frontend/src/components/
├── Dashboard.js                  # Main dashboard (MODIFIED: added button)
├── Dashboard.css                 # Dashboard styles (MODIFIED: button CSS)
├── AlgorithmPreviewModal.js      # Preview modal component (NEW)
└── AlgorithmPreviewModal.css     # Modal styles (NEW)
```

## Dependencies

### Backend
- Flask (already installed)
- Flask-CORS (already installed)
- Python 3.9+ (already required)

No new dependencies needed!

### Frontend
- React 18.2.0 (already installed)
- No new npm packages required!

## Troubleshooting

### Modal doesn't open
- Check browser console for errors
- Verify API is running on http://localhost:5000
- Check CORS configuration in backend

### No files appear in list
- Verify `backend/generate_algo/` directory exists
- Run a simulation to generate algorithm files
- Check backend logs for errors
- Try refreshing (polls every 5s automatically)

### Download doesn't work
- Check popup blocker settings
- Verify file exists on server
- Try copying and pasting code instead

### CORS errors
- Ensure Flask-CORS is installed: `pip install flask-cors`
- Verify `CORS(app)` is called in `app.py`
- Check browser console for specific CORS error

### Tests fail
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.9+)

## Future Enhancements

Potential improvements for future versions:

- [ ] Syntax highlighting in code preview
- [ ] Algorithm comparison view (side-by-side)
- [ ] Export as PDF or formatted document
- [ ] Filter/search by model or date
- [ ] Performance metrics per algorithm
- [ ] Edit and re-upload capability
- [ ] Version history tracking
- [ ] Share algorithm via URL

## Support

For issues or questions:
- Check this documentation
- Review backend logs: `backend/app.py` output
- Check browser console: F12 → Console tab
- Verify file permissions on `backend/generate_algo/`

## License

Same as the main AlgoClash project.
