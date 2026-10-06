import os

# app.py reads the key at import time; tests never hit the network.
os.environ.setdefault("LSE_API_KEY", "test-key")
