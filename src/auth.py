"""
src/auth.py
User authentication backend for the login page.

Stores users in a small SQLite table with salted **PBKDF2-HMAC-SHA256** password
hashes (standard library only — no plaintext passwords, no external deps). Used
by both the Streamlit login gate and the Gradio native login.

A default admin account is bootstrapped on first run from config
(DEEPFAKE_ADMIN_USER / DEEPFAKE_ADMIN_PASSWORD) so the app is usable out of the
box — change those in production.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.config import ADMIN_PASSWORD, ADMIN_USER, AUTH_DB
from src.logger import get_logger

logger = get_logger(__name__)

_PBKDF2_ROUNDS = 200_000


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Return (hex_hash, hex_salt) for a password using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return dk.hex(), salt


def verify_password(password: str, pw_hash: str, salt: str) -> bool:
    """Constant-time check of a password against a stored hash+salt."""
    calc, _ = hash_password(password, salt)
    return hmac.compare_digest(calc, pw_hash)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT UNIQUE NOT NULL,
    pw_hash    TEXT NOT NULL,
    salt       TEXT NOT NULL,
    role       TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL
);
"""


class AuthDB:
    """SQLite-backed user store."""

    def __init__(self, db_path: Path = AUTH_DB, bootstrap_admin: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        if bootstrap_admin:
            self._bootstrap_admin()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _bootstrap_admin(self) -> None:
        if self.user_exists(ADMIN_USER):
            return
        self.add_user(ADMIN_USER, ADMIN_PASSWORD, role="admin")
        logger.warning("Bootstrapped default admin '%s'. CHANGE the password via "
                       "DEEPFAKE_ADMIN_PASSWORD in production.", ADMIN_USER)

    # -- writes --------------------------------------------------------------
    def add_user(self, username: str, password: str, role: str = "user") -> bool:
        """Create a user. Returns False if the username already exists."""
        username = username.strip()
        if not username or not password:
            raise ValueError("Username and password are required.")
        pw_hash, salt = hash_password(password)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (username, pw_hash, salt, role, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (username, pw_hash, salt, role,
                     datetime.now().isoformat(timespec="seconds")),
                )
            logger.info("Created user '%s' (role=%s)", username, role)
            return True
        except sqlite3.IntegrityError:
            return False

    def set_password(self, username: str, password: str) -> bool:
        """Update a user's password. Returns False if the user doesn't exist."""
        pw_hash, salt = hash_password(password)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET pw_hash = ?, salt = ? WHERE username = ?",
                (pw_hash, salt, username.strip()),
            )
        return cur.rowcount > 0

    def delete_user(self, username: str) -> bool:
        """Delete a user. Returns False if the user doesn't exist."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM users WHERE username = ?",
                               (username.strip(),))
        if cur.rowcount > 0:
            logger.info("Deleted user '%s'", username)
        return cur.rowcount > 0

    def count_by_role(self, role: str) -> int:
        """Number of users with a given role (used to protect the last admin)."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM users WHERE role = ?",
                               (role,)).fetchone()
        return int(row[0])

    # -- reads ---------------------------------------------------------------
    def user_exists(self, username: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE username = ?",
                               (username.strip(),)).fetchone()
        return row is not None

    def verify_credentials(self, username: str, password: str) -> bool:
        """Return True iff the username exists and the password matches."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pw_hash, salt FROM users WHERE username = ?",
                (username.strip(),)).fetchone()
        if row is None:
            return False
        return verify_password(password, row["pw_hash"], row["salt"])

    def get_role(self, username: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT role FROM users WHERE username = ?",
                               (username.strip(),)).fetchone()
        return row["role"] if row else None

    def list_users(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT username, role, created_at FROM users ORDER BY id").fetchall()
        return [dict(r) for r in rows]


# Module-level singleton.
_auth_db: Optional[AuthDB] = None


def get_auth_db() -> AuthDB:
    """Return a shared AuthDB instance (bootstraps the admin on first use)."""
    global _auth_db
    if _auth_db is None:
        _auth_db = AuthDB()
    return _auth_db


def authenticate(username: str, password: str) -> bool:
    """Convenience wrapper used by the Streamlit and Gradio login flows."""
    return get_auth_db().verify_credentials(username, password)
