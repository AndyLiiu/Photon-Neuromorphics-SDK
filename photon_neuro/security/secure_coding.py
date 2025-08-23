"""
Secure Coding Utilities
=======================

Utilities for secure coding practices.
"""

import hashlib
import secrets
import base64
from typing import Optional

def secure_hash(data: str, algorithm: str = "sha256") -> str:
    """Generate secure hash of data."""
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "sha3_256":
        hasher = hashlib.sha3_256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hasher.update(data.encode('utf-8'))
    return hasher.hexdigest()

def secure_random(length: int = 32) -> str:
    """Generate cryptographically secure random string."""
    return secrets.token_hex(length)

def secure_random_bytes(length: int = 32) -> bytes:
    """Generate cryptographically secure random bytes."""
    return secrets.token_bytes(length)

def encrypt_data(data: str, key: Optional[str] = None) -> str:
    """Simple encryption using Fernet (if available)."""
    try:
        from cryptography.fernet import Fernet
        
        if key is None:
            # Generate a new key
            key = Fernet.generate_key()
        elif isinstance(key, str):
            key = key.encode()
        
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
        
    except ImportError:
        # Fallback to base64 encoding (NOT SECURE)
        return base64.b64encode(data.encode()).decode()

def decrypt_data(encrypted_data: str, key: str) -> str:
    """Simple decryption using Fernet (if available)."""
    try:
        from cryptography.fernet import Fernet
        
        if isinstance(key, str):
            key = key.encode()
        
        f = Fernet(key)
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = f.decrypt(encrypted_bytes)
        return decrypted.decode()
        
    except ImportError:
        # Fallback to base64 decoding (NOT SECURE)
        return base64.b64decode(encrypted_data.encode()).decode()

def constant_time_compare(a: str, b: str) -> bool:
    """Compare strings in constant time to prevent timing attacks."""
    return secrets.compare_digest(a, b)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    import os
    
    # Remove path separators and dangerous characters
    sanitized = filename.replace(os.sep, "_").replace(os.altsep or "", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in "._-")
    
    # Limit length
    sanitized = sanitized[:255]
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "file"
    
    return sanitized

def validate_input_length(data: str, max_length: int = 1000) -> str:
    """Validate input length to prevent DoS attacks."""
    if len(data) > max_length:
        raise ValueError(f"Input length {len(data)} exceeds maximum {max_length}")
    return data