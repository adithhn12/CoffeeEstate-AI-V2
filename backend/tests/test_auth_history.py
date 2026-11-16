import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend root is on path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app
from app.database import SessionLocal
from app.models.user import User
from app.models.analysis import ImageAnalysis

client = TestClient(app)

def test_auth_and_history():
    print("--- 1. Testing Registration ---")
    test_email = "testuser@coffee.com"
    test_user = "coffeefarmer"
    test_pass = "Password123"

    # Clean up test user if already exists from previous runs
    with SessionLocal() as db:
        existing = db.query(User).filter(User.email == test_email).first()
        if existing:
            db.delete(existing)
            db.commit()

    reg_resp = client.post("/api/auth/register", json={
        "username": test_user,
        "email": test_email,
        "password": test_pass
    })
    assert reg_resp.status_code == 201, f"Registration failed: {reg_resp.text}"
    data = reg_resp.json()
    assert "access_token" in data
    assert data["user"]["email"] == test_email
    token = data["access_token"]
    user_id = data["user"]["id"]
    print("Registration OK! Token received.")

    print("--- 2. Testing Login ---")
    login_resp = client.post("/api/auth/login", json={
        "username_or_email": test_email,
        "password": test_pass
    })
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    login_token = login_resp.json()["access_token"]
    assert login_token is not None

    bad_login = client.post("/api/auth/login", json={
        "username_or_email": test_email,
        "password": "wrongpassword"
    })
    assert bad_login.status_code == 401
    print("Login & Bad Login verification OK!")

    print("--- 3. Testing Current User Profile ---")
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == test_user
    print("Profile endpoint OK!")

    print("--- 4. Testing History as Guest (Rule 1) ---")
    # Guest fetching history should receive an empty list
    guest_history = client.get("/api/analysis/history")
    assert guest_history.status_code == 200
    assert guest_history.json() == []
    print("Guest history is empty as expected for unauthenticated users.")

    print("--- 5. Testing History as Authenticated User (Rule 2) ---")
    user_history = client.get("/api/analysis/history", headers={"Authorization": f"Bearer {token}"})
    assert user_history.status_code == 200
    print(f"Authenticated user history fetched successfully ({len(user_history.json())} items).")

    print("\nALL BACKEND AUTH & HISTORY TESTS PASSED!")

if __name__ == "__main__":
    test_auth_and_history()
