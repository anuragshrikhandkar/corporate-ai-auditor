import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
