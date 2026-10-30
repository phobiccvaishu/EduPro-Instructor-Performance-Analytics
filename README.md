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

## Documentation & Architecture

- Architecture Guide: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- UI & Design System Guide: [docs/UI_GUIDE.md](docs/UI_GUIDE.md)
- Known Issues & Roadmap: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
- Visual Artifacts & Screenshots: [docs/screenshots/](docs/screenshots/)

## Running Tests & Linting

```bash
# Run complete test suite
pytest

# Run linting check
ruff check .

# Check code formatting
ruff format --check .
```
```
