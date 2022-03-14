import os

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)