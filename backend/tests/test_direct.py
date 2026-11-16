import os
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.database import SessionLocal
from app.models.user import User
from app.models.analysis import ImageAnalysis
from app.schemas.auth import UserRegisterRequest, UserLoginRequest
from app.routes.auth import register, login, get_current_user_profile
from app.routes.analysis import get_analysis_history

def test_direct():
    with SessionLocal() as db:
        test_email = "testrunner@coffee.com"
        test_user = "farmerbob"
        test_pass = "SecurePass123"

        # Cleanup existing
        existing = db.query(User).filter(User.email == test_email).first()
        if existing:
            db.delete(existing)
            db.commit()

        # 1. Register
        reg_req = UserRegisterRequest(username=test_user, email=test_email, password=test_pass)
        token_resp = register(reg_req, db)
        print("Registration successful! Token:", token_resp.access_token[:20], "...")
        assert token_resp.user.username == test_user

        # 2. Login
        login_req = UserLoginRequest(username_or_email=test_email, password=test_pass)
        login_res = login(login_req, db)
        print("Login successful for user id:", login_res.user.id)
        assert login_res.user.email == test_email

        # 3. History for Guest
        guest_hist = get_analysis_history(section_id=None, limit=50, current_user=None, db=db)
        print("Guest history count:", len(guest_hist))
        assert guest_hist == []

        # 4. History for Logged in user
        user = db.query(User).filter(User.id == token_resp.user.id).first()
        user_hist = get_analysis_history(section_id=None, limit=50, current_user=user, db=db)
        print("Logged-in user history count:", len(user_hist))

        print("\n--> Direct backend auth & history verification SUCCESSFUL!")

if __name__ == "__main__":
    test_direct()
