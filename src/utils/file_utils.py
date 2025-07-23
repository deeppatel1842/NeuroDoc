"""
File Utilities

This module contains utilities for file operations, directory management,
and path handling for the NeuroDoc system.
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object of the created/existing directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # Remove or replace problematic characters
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '.-_':
            safe_chars.append(char)
        elif char in ' /\\':
            safe_chars.append('_')
    
    safe_name = ''.join(safe_chars)
    
    # Ensure filename isn't too long
    if len(safe_name) > 200:
        name_part, ext = os.path.splitext(safe_name)
        safe_name = name_part[:200-len(ext)] + ext
    
    return safe_name


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Get SHA256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash string
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_content_hash(content: bytes) -> str:
    """
    Get SHA256 hash of content bytes.
    
    Args:
        content: Content bytes
        
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(content).hexdigest()


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2):
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: JSON file path
        
    Returns:
        Loaded data dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return list(directory.glob(pattern))


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def delete_file_safe(file_path: Union[str, Path]) -> bool:
    """
    Safely delete a file, handling errors gracefully.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def delete_directory_safe(dir_path: Union[str, Path]) -> bool:
    """
    Safely delete a directory and all its contents.
    
    Args:
        dir_path: Path to directory to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        dir_path = Path(dir_path)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            logger.info(f"Deleted directory: {dir_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete directory {dir_path}: {e}")
        return False


def copy_file_safe(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Safely copy a file, handling errors gracefully.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src, dst = Path(src), Path(dst)
        ensure_directory(dst.parent)
        shutil.copy2(src, dst)
        logger.info(f"Copied file: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy file {src} -> {dst}: {e}")
        return False


def get_directory_size(dir_path: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        dir_path: Directory path
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    dir_path = Path(dir_path)
    
    if not dir_path.exists():
        return 0
    
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                continue
    
    return total_size


def cleanup_old_files(
    directory: Union[str, Path], 
    max_age_days: int = 30,
    pattern: str = "*"
) -> int:
    """
    Clean up old files in a directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    import time
    
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    deleted_count = 0
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            try:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete old file {file_path}: {e}")
                continue
    
    return deleted_count


class FileValidator:
    """Utility class for file validation."""
    
    ALLOWED_EXTENSIONS = {'.pdf'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def validate_file_type(cls, filename: str) -> bool:
        """
        Validate file type by extension.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if valid, False otherwise
        """
        extension = Path(filename).suffix.lower()
        return extension in cls.ALLOWED_EXTENSIONS
    
    @classmethod
    def validate_file_size(cls, size: int) -> bool:
        """
        Validate file size.
        
        Args:
            size: File size in bytes
            
        Returns:
            True if valid, False otherwise
        """
        return 0 < size <= cls.MAX_FILE_SIZE
    
    @classmethod
    def validate_content_type(cls, content_type: str) -> bool:
        """
        Validate MIME content type.
        
        Args:
            content_type: MIME type string
            
        Returns:
            True if valid, False otherwise
        """
        allowed_types = {
            'application/pdf',
            'application/x-pdf'
        }
        return content_type.lower() in allowed_types
