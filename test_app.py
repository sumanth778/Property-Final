# test_app.py

from app import app

def test_home_route():
    tester = app.test_client()
    response = tester.get('/')
    assert response.status_code == 200
    assert b"Hello" in response.data or b"<" in response.data