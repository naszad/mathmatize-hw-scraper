# MathMatize Homework Scraper

A Python script to interact with MathMatize, fetch assignments, and optionally auto-answer exercises.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```env
MATHMATIZE_USERNAME=your_email
MATHMATIZE_PASSWORD=your_password

## Usage

Basic usage:
```bash
python mathmatize_scraper.py
```

### Command Line Flags

The script supports several command-line flags to modify its behavior:

| Flag | Description |
|------|-------------|
| `--auto-answer` | Enable interactive auto-answer mode. You'll be prompted before answering each exercise. |
| `--force` | When used with `--auto-answer`, automatically answers all exercises without prompting. |
| `--verbose` | Enable detailed output showing encryption/decryption process and other details. |
| `--fast` | When used with `--verbose`, disables output delays for faster execution. |

### Examples

1. Basic run (just fetch assignments and exercises):
```bash
python mathmatize_scraper.py
```

2. Interactive auto-answer mode:
```bash
python mathmatize_scraper.py --auto-answer
```

3. Automatic answer mode (no prompts):
```bash
python mathmatize_scraper.py --auto-answer --force
```

4. Debug mode with detailed output:
```bash
python mathmatize_scraper.py --verbose
```

5. Fast debug mode:
```bash
python mathmatize_scraper.py --verbose --fast
```

## Output

The script creates an `outputs` directory containing:
- `mathmatize_assignments_YYYYMMDD_HHMMSS.txt`: List of assignments
- `mathmatize_exercises_YYYYMMDD_HHMMSS.json`: Detailed exercise data
- `mathmatize_scraper_YYYYMMDD_HHMMSS.log`: Script execution log