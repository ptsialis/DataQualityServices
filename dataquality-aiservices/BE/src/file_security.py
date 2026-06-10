"""
File Upload Security Module
Prevents CSV injection, ARFF injection, and other malicious payload attacks.
"""

import re
import pandas as pd
import logging
from typing import Tuple, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# CSV Injection patterns - MORE SPECIFIC to avoid false positives on numeric data
# Only flag actual formula/command injection, not legitimate numeric values
CSV_INJECTION_PATTERNS = [
    r'^=[^0-9]',           # = followed by non-digit (formulas like =SUM, =CMD)
    r'^\+[^0-9]',          # + followed by non-digit (like +2+5+cmd, but not +123)
    r'^-[a-zA-Z@!]',       # - followed by letter/@ (like -cmd, but not -189.32)
    r'^@',                 # @ at start (like @SUM, @import)
    r'^\|.*\|',            # Pipe commands |cmd|
    r'^%.*%',              # Percent-encoded commands %cmd%
]

# ARFF Injection patterns
ARFF_INJECTION_PATTERNS = [
    r'@\s*relation\s+[^{]*{',  # Nested relations (potential code injection)
    r'@\s*attribute.*{.*}',      # Complex attribute definitions
]

# File size limits
MAX_CSV_SIZE_MB = 500
MAX_ARFF_SIZE_MB = 500
MAX_JSON_SIZE_MB = 500
MAX_PARQUET_SIZE_MB = 1000

FILE_SIZE_LIMITS = {
    'csv': MAX_CSV_SIZE_MB * 1024 * 1024,
    'txt': MAX_CSV_SIZE_MB * 1024 * 1024,
    'tsv': MAX_CSV_SIZE_MB * 1024 * 1024,
    'arff': MAX_ARFF_SIZE_MB * 1024 * 1024,
    'json': MAX_JSON_SIZE_MB * 1024 * 1024,
    'parquet': MAX_PARQUET_SIZE_MB * 1024 * 1024,
    'arrow': MAX_PARQUET_SIZE_MB * 1024 * 1024,
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'csv', 'txt', 'tsv', 'arff', 'json', 'jsonl',
    'parquet', 'pq', 'arrow', 'feather'
}

# Dangerous characters for filenames
DANGEROUS_FILENAME_CHARS = r'[<>:"|?*\x00-\x1f]'


def validate_filename(filename: str) -> Tuple[bool, str]:
    """
    Validate filename for security issues.
    Returns (is_valid, error_message)
    """
    if not filename:
        return False, "Dateiname ist leer."
    
    # Check length
    if len(filename) > 255:
        return False, "Dateiname ist zu lang (max 255 Zeichen)."
    
    # Check for dangerous characters
    if re.search(DANGEROUS_FILENAME_CHARS, filename):
        return False, "Dateiname enthält ungültige Zeichen."
    
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Dateiname enthält ungültige Pfad-Sequenzen."
    
    # Extract extension
    ext = Path(filename).suffix.lower().lstrip('.')
    
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ', '.join(sorted(ALLOWED_EXTENSIONS))
        return False, f"Dateityp nicht erlaubt. Erlaubte Typen: {allowed}"
    
    return True, ""


def validate_file_size(file_path: str, file_ext: str) -> Tuple[bool, str]:
    """
    Validate file size based on type.
    Returns (is_valid, error_message)
    """
    try:
        file_size = Path(file_path).stat().st_size
    except OSError as e:
        return False, f"Fehler beim Lesen der Datei: {e}"
    
    max_size = FILE_SIZE_LIMITS.get(file_ext.lower(), 100 * 1024 * 1024)
    
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"Datei zu groß (max {max_mb:.0f} MB)."
    
    if file_size == 0:
        return False, "Datei ist leer."
    
    return True, ""


def sanitize_cell_value(value: Any) -> Any:
    """
    Sanitize a single cell value to prevent formula injection.
    Only flags actual injection attempts, not legitimate numeric values.
    """
    if value is None or pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    if not value_str:
        return value
    
    # Check for formula injection patterns (more specific to avoid false positives)
    # = followed by non-digit (=SUM, =CMD, etc.) - but allow =123
    if value_str.startswith('=') and len(value_str) > 1 and not value_str[1].isdigit():
        logger.warning(f"Formula injection detected: {value_str[:50]}")
        return "'" + value_str
    
    # + followed by non-digit (+2+5+cmd, but not +123)
    if value_str.startswith('+') and len(value_str) > 1 and not value_str[1].isdigit():
        logger.warning(f"Formula injection detected: {value_str[:50]}")
        return "'" + value_str
    
    # - followed by letter/@ (like -cmd, -@import), but allow negative numbers (-189.32)
    if value_str.startswith('-') and len(value_str) > 1:
        next_char = value_str[1]
        # Only flag if next char is letter or @, not digit or dot
        if next_char.isalpha() or next_char == '@':
            logger.warning(f"Formula injection detected: {value_str[:50]}")
            return "'" + value_str
    
    # @ at start (like @SUM, @import)
    if value_str.startswith('@'):
        logger.warning(f"Formula injection detected: {value_str[:50]}")
        return "'" + value_str
    
    # Pipe commands |cmd|
    if value_str.startswith('|') and value_str.endswith('|'):
        logger.warning(f"Pipe command injection detected: {value_str[:50]}")
        return value_str
    
    # Percent-encoded commands %cmd%
    if value_str.startswith('%') and value_str.endswith('%'):
        logger.warning(f"Percent-encoded command injection detected: {value_str[:50]}")
        return value_str
    
    return value


def sanitize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Sanitize DataFrame cells to prevent injection attacks.
    Returns (sanitized_df, warnings_list)
    """
    warnings = []
    df_safe = df.copy()
    
    for col in df_safe.columns:
        for idx, value in enumerate(df_safe[col]):
            sanitized = sanitize_cell_value(value)
            if sanitized != value:
                warnings.append(
                    f"Cell [{idx}, '{col}']: Injection pattern removed"
                )
                df_safe.at[idx, col] = sanitized
    
    if warnings:
        logger.warning(f"Sanitization completed with {len(warnings)} warnings")
    
    return df_safe, warnings


def validate_csv_content(file_path: str, sample_rows: int = 100) -> Tuple[bool, str, List[str]]:
    """
    Validate CSV content for injection attacks.
    Returns (is_valid, error_message, warnings_list)
    """
    warnings = []
    
    try:
        # Read sample to check for injection patterns
        df_sample = pd.read_csv(file_path, nrows=sample_rows, on_bad_lines='skip')
        
        # Check each cell in sample
        for col in df_sample.columns:
            for idx, value in enumerate(df_sample[col]):
                if pd.isna(value):
                    continue
                
                value_str = str(value).strip()
                
                # Check CSV injection patterns
                for pattern in CSV_INJECTION_PATTERNS:
                    if re.match(pattern, value_str):
                        warnings.append(
                            f"Potentielle CSV-Injection in Zeile {idx}, Spalte '{col}': "
                            f"'{value_str[:50]}...'"
                        )
                        logger.warning(f"CSV injection pattern detected: {pattern}")
        
        return True, "", warnings
    
    except Exception as e:
        return False, f"Fehler beim Validieren der CSV: {str(e)}", []


def validate_arff_content(file_path: str) -> Tuple[bool, str, List[str]]:
    """
    Validate ARFF content for injection attacks.
    Returns (is_valid, error_message, warnings_list)
    """
    warnings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Check for ARFF injection patterns
        for pattern in ARFF_INJECTION_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                warnings.append(
                    f"Potentielle ARFF-Injection erkannt: {match.group()[:100]}"
                )
                logger.warning(f"ARFF injection pattern detected: {pattern}")
        
        # Check for suspicious keywords in attributes
        suspicious_keywords = ['exec', 'eval', 'system', 'shell', 'bash', '__import__']
        for keyword in suspicious_keywords:
            if keyword.lower() in content.lower():
                warnings.append(f"Verdächtiges Schlüsselwort erkannt: {keyword}")
                logger.warning(f"Suspicious keyword in ARFF: {keyword}")
        
        return True, "", warnings
    
    except Exception as e:
        return False, f"Fehler beim Validieren der ARFF: {str(e)}", []


def validate_json_content(file_path: str) -> Tuple[bool, str, List[str]]:
    """
    Validate JSON content for injection attacks.
    Returns (is_valid, error_message, warnings_list)
    """
    warnings = []
    
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check for suspicious patterns in JSON values
        def check_json_value(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str) and any(suspicious in key.lower() for suspicious in ['eval', 'exec', '__']):
                        warnings.append(f"Verdächtiger Schlüssel: {path}.{key}")
                    check_json_value(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    check_json_value(item, f"{path}[{idx}]")
            elif isinstance(obj, str):
                # Check for formula injection in strings
                if obj.strip().startswith(('=', '+', '-', '@')):
                    warnings.append(f"Potentielle Injection in {path}: {obj[:50]}")
        
        check_json_value(data)
        return True, "", warnings
    
    except json.JSONDecodeError as e:
        return False, f"Ungültiges JSON-Format: {str(e)}", []
    except Exception as e:
        return False, f"Fehler beim Validieren der JSON: {str(e)}", []


def validate_uploaded_file(
    file_path: str,
    filename: str,
    file_ext: str
) -> Tuple[bool, str, List[str], pd.DataFrame]:
    """
    Complete validation of uploaded file.
    Returns (is_valid, error_message, warnings_list, sanitized_df)
    """
    all_warnings = []
    
    # 1. Validate filename
    is_valid, error = validate_filename(filename)
    if not is_valid:
        return False, error, [], None
    
    # 2. Validate file size
    is_valid, error = validate_file_size(file_path, file_ext)
    if not is_valid:
        return False, error, [], None
    
    # 3. Format-specific validation
    if file_ext.lower() == 'csv':
        is_valid, error, warnings = validate_csv_content(file_path)
        if not is_valid:
            return False, error, [], None
        all_warnings.extend(warnings)
    
    elif file_ext.lower() == 'arff':
        is_valid, error, warnings = validate_arff_content(file_path)
        if not is_valid:
            return False, error, [], None
        all_warnings.extend(warnings)
    
    elif file_ext.lower() == 'json':
        is_valid, error, warnings = validate_json_content(file_path)
        if not is_valid:
            return False, error, [], None
        all_warnings.extend(warnings)
    
    return True, "", all_warnings, None


def log_security_event(event_type: str, filename: str, details: str = ""):
    """
    Log security-related events for audit trail.
    """
    logger.warning(
        f"[SECURITY] {event_type} | File: {filename} | {details}"
    )
