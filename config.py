import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TARGET_REPETITIONS = 6

RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "results")
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
VIDEOS_FOLDER = os.path.join(DATA_FOLDER, "videos")

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
