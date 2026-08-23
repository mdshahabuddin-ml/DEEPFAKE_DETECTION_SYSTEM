"""Tests for the authentication backend."""

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


def test_admin_bootstrapped(tmp_path):
    db = AuthDB(db_path=tmp_path / "users.db")
    assert db.user_exists("admin")
    assert db.verify_credentials("admin", "admin123") is True
    assert db.get_role("admin") == "admin"


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
