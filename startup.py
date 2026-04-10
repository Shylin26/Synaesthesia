import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from engine.db import setup_db
from engine.emotion_tracker import setup_tracker

setup_db()
setup_tracker()
print("Database initialized.")
