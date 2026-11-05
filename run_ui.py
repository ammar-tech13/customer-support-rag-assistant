"""Run the Streamlit UI."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    streamlit_app = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(streamlit_app)
    ])
