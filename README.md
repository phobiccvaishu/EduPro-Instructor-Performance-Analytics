# EduPro Analytics — Instructor Performance & Course Quality Evaluation

Streamlit analytics platform evaluating instructor performance, course quality, and student enrollment trends on the EduPro education platform.

## Setup

Ensure Python 3.10+ is installed.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install production dependencies
pip install -r requirements.txt

# 3. (Optional) Install development and testing dependencies
pip install -r requirements-dev.txt
```

## Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Optional: override data file location using the `EDUPRO_DATA_PATH` environment variable:
```bash
# PowerShell
$env:EDUPRO_DATA_PATH = "data/raw/EduPro_Online_Platform.xlsx"
streamlit run app.py
```

## Running Tests & Linting

```bash
# Run complete test suite (20 tests)
pytest

# Run linting check
ruff check
```
