"""
Flask blueprint for algorithm preview and management endpoints.
Provides secure access to generated algorithm files in backend/generate_algo/.
"""

import os
import re
from pathlib import Path
from datetime import datetime
from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

algos_bp = Blueprint('algos', __name__, url_prefix='/api/algos')

# Define the output directory for generated algorithms
BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_ROOT / "generate_algo"

# Whitelist pattern for valid algorithm filenames
VALID_FILENAME_PATTERN = re.compile(r'^generated_algo_[a-zA-Z0-9_\-]+\.py$')


def is_safe_filename(filename):
    """
    Validate filename against whitelist pattern and check for directory traversal.
    Only allows filenames matching: generated_algo_*.py
    """
    if not filename or not isinstance(filename, str):
        return False

    # Check whitelist pattern
    if not VALID_FILENAME_PATTERN.match(filename):
        return False

    # Prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return False

    return True


def safe_path(filename):
    """
    Construct and validate a safe file path within OUTPUT_DIR.
    Returns (is_valid, full_path) tuple.
    """
    if not is_safe_filename(filename):
        return False, None

    # Construct path and resolve to absolute
    file_path = (OUTPUT_DIR / filename).resolve()
    output_dir_resolved = OUTPUT_DIR.resolve()

    # Ensure the resolved path is within OUTPUT_DIR
    try:
        file_path.relative_to(output_dir_resolved)
    except ValueError:
        # Path is outside OUTPUT_DIR
        return False, None

    return True, file_path


def get_model_name_from_filename(filename):
    """
    Extract model name from filename.
    Example: generated_algo_gpt-4.py -> gpt-4
    """
    match = re.match(r'^generated_algo_(.+)\.py$', filename)
    return match.group(1) if match else filename


@algos_bp.route('', methods=['GET'])
def list_algorithms():
    """
    GET /api/algos
    List all generated algorithm files with metadata.
    Returns JSON array of file objects sorted by modified time (newest first).
    """
    try:
        if not OUTPUT_DIR.exists():
            return jsonify([])

        files = []
        for file_path in OUTPUT_DIR.glob('generated_algo_*.py'):
            if not file_path.is_file():
                continue

            filename = file_path.name
            if not is_safe_filename(filename):
                continue

            stat = file_path.stat()
            files.append({
                'filename': filename,
                'modelName': get_model_name_from_filename(filename),
                'sizeBytes': stat.st_size,
                'createdAt': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modifiedAt': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        # Sort by modified time descending (newest first)
        files.sort(key=lambda x: x['modifiedAt'], reverse=True)

        return jsonify(files)

    except Exception as e:
        print(f"Error listing algorithms: {e}")
        return jsonify({'error': 'Failed to list algorithms'}), 500


@algos_bp.route('/<filename>', methods=['GET'])
def get_algorithm(filename):
    """
    GET /api/algos/:filename
    Returns the raw text content of a specific algorithm file.
    Content-Type: text/plain; charset=utf-8
    """
    try:
        is_valid, file_path = safe_path(filename)
        if not is_valid:
            return jsonify({'error': 'Invalid filename'}), 400

        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404

        # Read and return file content as plain text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    except Exception as e:
        print(f"Error reading algorithm {filename}: {e}")
        return jsonify({'error': 'Failed to read algorithm'}), 500


@algos_bp.route('/<filename>/download', methods=['GET'])
def download_algorithm(filename):
    """
    GET /api/algos/:filename/download
    Download algorithm file as attachment.
    """
    try:
        is_valid, file_path = safe_path(filename)
        if not is_valid:
            return jsonify({'error': 'Invalid filename'}), 400

        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404

        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/plain'
        )

    except Exception as e:
        print(f"Error downloading algorithm {filename}: {e}")
        return jsonify({'error': 'Failed to download algorithm'}), 500


@algos_bp.route('/<filename>', methods=['DELETE'])
def delete_algorithm(filename):
    """
    DELETE /api/algos/:filename
    Delete a single algorithm file.
    """
    try:
        is_valid, file_path = safe_path(filename)
        if not is_valid:
            return jsonify({'error': 'Invalid filename'}), 400

        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404

        file_path.unlink()
        return jsonify({'deleted': True, 'filename': filename}), 200

    except Exception as e:
        print(f"Error deleting algorithm {filename}: {e}")
        return jsonify({'error': 'Failed to delete algorithm'}), 500


@algos_bp.route('', methods=['DELETE'])
def delete_all_algorithms():
    """
    DELETE /api/algos
    Delete all generated algorithm files (bulk delete).
    """
    try:
        if not OUTPUT_DIR.exists():
            return jsonify({'deleted': 0, 'message': 'No algorithms directory found'}), 200

        deleted_count = 0
        for file_path in OUTPUT_DIR.glob('generated_algo_*.py'):
            if file_path.is_file() and is_safe_filename(file_path.name):
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path.name}: {e}")

        return jsonify({
            'deleted': deleted_count,
            'message': f'Deleted {deleted_count} algorithm file(s)'
        }), 200

    except Exception as e:
        print(f"Error deleting all algorithms: {e}")
        return jsonify({'error': 'Failed to delete algorithms'}), 500
