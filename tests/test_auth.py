"""Tests for the authentication backend."""

import os

from src.auth import AuthDB, hash_password, verify_password


def test_password_hash_roundtrip():
    pw_hash, salt = hash_password("s3cret!")
    assert pw_hash != "s3cret!"                 # never store plaintext
    assert verify_password("s3cret!", pw_hash, salt) is True
    assert verify_password("wrong", pw_hash, salt) is False


def test_salt_is_random():
    h1, s1 = hash_password("same")
    h2, s2 = hash_password("same")
    assert s1 != s2 and h1 != h2                # different salts -> different hashes


def test_admin_bootstrapped_with_random_password(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPFAKE_ADMIN_PASSWORD", raising=False)
    db = AuthDB(db_path=tmp_path / "users.db")
    assert db.user_exists("admin")
    assert db.get_role("admin") == "admin"
    # The weak default must never be usable: bootstrap generates a random one.
    assert db.verify_credentials("admin", "admin123") is False


def test_admin_bootstrapped_with_strong_env_password(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ADMIN_PASSWORD", "correct-horse-battery")
    db = AuthDB(db_path=tmp_path / "users.db")
    assert db.verify_credentials("admin", "correct-horse-battery") is True


def test_admin_bootstrap_refuses_weak_password_in_production(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ADMIN_PASSWORD", "admin123")
    monkeypatch.setenv("DEEPFAKE_ENV", "production")
    import pytest
    with pytest.raises(RuntimeError, match="weak"):
        AuthDB(db_path=tmp_path / "users.db")


def test_add_and_authenticate(tmp_path):
    db = AuthDB(db_path=tmp_path / "users.db")
    assert db.add_user("analyst", "pw12345") is True
    assert db.add_user("analyst", "other") is False       # duplicate rejected
    assert db.verify_credentials("analyst", "pw12345") is True
    assert db.verify_credentials("analyst", "nope") is False
    assert db.verify_credentials("ghost", "x") is False    # unknown user


def test_set_password(tmp_path):
    db = AuthDB(db_path=tmp_path / "users.db")
    db.add_user("u", "old")
    assert db.set_password("u", "new") is True
    assert db.verify_credentials("u", "new") is True
    assert db.verify_credentials("u", "old") is False
    assert db.set_password("ghost", "x") is False


def test_delete_user_and_role_count(tmp_path):
    db = AuthDB(db_path=tmp_path / "users.db")
    assert db.count_by_role("admin") == 1          # bootstrapped admin
    db.add_user("temp", "pw")
    assert db.delete_user("temp") is True
    assert db.user_exists("temp") is False
    assert db.delete_user("temp") is False         # already gone

    db.add_user("admin2", "pw", role="admin")
    assert db.count_by_role("admin") == 2
