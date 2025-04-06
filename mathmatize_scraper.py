#!/usr/bin/env python3
"""
mathmatize_scraper.py (COMPLEX STYLE)
Author: Possibly You

PURPOSE:
  1) Authenticate with MathMatize
  2) Fetch assignments & exercises
  3) Optionally auto-answer them
  4) Print TONS of verbose encryption/decryption logs
     and show fancy progress bars for visual effect
  5) Store data in PostgreSQL database
"""

import os
import re
import json
import base64
import logging
import sys
import time
import random
from datetime import datetime, date
from dotenv import load_dotenv
import requests
from tqdm import tqdm
from collections import defaultdict # Import defaultdict

# Import psycopg2 if --db is set
if "--db" in sys.argv:
    try:
        import psycopg2
        import psycopg2.extras
        from psycopg2.extras import Json
    except ImportError as e:
        print("Error importing psycopg2:", str(e))
        print("Please make sure PostgreSQL is installed and try:")
        print("pip uninstall psycopg2")
        print("pip install psycopg2-binary")
        sys.exit(1)

def get_connection():
    try:
        return psycopg2.connect(
            host="10.6.254.67",
            database="mathmatize",
            user="postgres",
            password="postgres",
            port="5432"
        )
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def init_database():
    conn = get_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fb_req (
                    id SERIAL PRIMARY KEY,
                    fbname VARCHAR NOT NULL,
                    fbkey VARCHAR NOT NULL,
                    firebase_api_key VARCHAR,
                    id_token VARCHAR,
                    jwt_token VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS assignments (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR NOT NULL,
                    description TEXT,
                    open_date TIMESTAMP,
                    due_date TIMESTAMP,
                    points INTEGER,
                    points_earned INTEGER,
                    completion VARCHAR,
                    status VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exercises (
                    id VARCHAR PRIMARY KEY,
                    assignment_id VARCHAR REFERENCES assignments(id),
                    raw_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False
    finally:
        conn.close()

def store_assignment(assignment_data):
    conn = get_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO assignments (
                    id, name, description, open_date, due_date,
                    points, points_earned, completion, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    open_date = EXCLUDED.open_date,
                    due_date = EXCLUDED.due_date,
                    points = EXCLUDED.points,
                    points_earned = EXCLUDED.points_earned,
                    completion = EXCLUDED.completion,
                    status = EXCLUDED.status
            """, (
                assignment_data["id"],
                assignment_data["name"],
                assignment_data["description"],
                assignment_data.get("open_date"),
                assignment_data.get("due_date"),
                assignment_data["points"],
                assignment_data["points_earned"],
                assignment_data["completion"],
                assignment_data["status"]
            ))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error storing assignment: {e}")
        return False
    finally:
        conn.close()

def store_exercise(exercise_data, assignment_id):
    conn = get_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO exercises (id, assignment_id, raw_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    raw_data = EXCLUDED.raw_data
            """, (
                exercise_data["id"],
                assignment_id,
                Json(exercise_data["raw_data"])
            ))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error storing exercise: {e}")
        return False
    finally:
        conn.close()

def store_fb_req(fbname, fbkey, firebase_api_key=None, id_token=None, jwt_token=None):
    conn = get_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fb_req (fbname, fbkey, firebase_api_key, id_token, jwt_token)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    fbname = EXCLUDED.fbname,
                    fbkey = EXCLUDED.fbkey,
                    firebase_api_key = EXCLUDED.firebase_api_key,
                    id_token = EXCLUDED.id_token,
                    jwt_token = EXCLUDED.jwt_token,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (fbname, fbkey, firebase_api_key, id_token, jwt_token))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error storing fb_req: {e}")
        return False
    finally:
        conn.close()

def get_fb_req():
    conn = get_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT fbname, fbkey, firebase_api_key, id_token, jwt_token
                FROM fb_req
                ORDER BY updated_at DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    "fbname": row[0],
                    "fbkey": row[1],
                    "firebase_api_key": row[2],
                    "id_token": row[3],
                    "jwt_token": row[4]
                }
            return None
    except Exception as e:
        logger.error(f"Error retrieving fb_req: {e}")
        return None
    finally:
        conn.close()

################################################################################
#                               F A K E   F X
################################################################################

# Global state for matrix mode
MATRIX_MODE = False  # Global matrix mode flag
MATRIX_DATA = []  # Store encrypted payloads that need decryption

# Delay constants (in seconds) – from slow (headers) to fast (raw data)
class Delays:
    def __init__(self, fast_mode=False):
        if fast_mode:
            self.HEADER = 0
            self.ENCRYPTED = 0.000001  # 1 microsecond - smallest practical delay
            self.FINAL_JSON = 0
            self.RAW = 0
        else:
            self.HEADER = 0.05    # For headers (printed slowest)
            self.ENCRYPTED = 0.005   # For encrypted data displays
            self.FINAL_JSON = 0.001   # For final parsed JSON outputs
            self.RAW = 0.00001 # For large raw unfiltered data (printed fastest)

DELAYS = Delays()  # Global delays object

def sanitize_output(text: str, is_json: bool = False) -> str:
    """
    Sanitize text for terminal output to prevent encoding issues.
    Handles escape sequences and special characters that might affect terminal display.
    
    Args:
        text: The text to sanitize
        is_json: If True, preserves JSON formatting without adding terminal resets
    """
    if not isinstance(text, str):
        text = str(text)
        
    # First encode as bytes then decode to handle any invalid sequences
    text = text.encode('utf-8', errors='replace').decode('utf-8')
    
    if is_json:
        # For JSON content, we only handle invalid characters without adding terminal resets
        replacements = {
            '\x1b': '',     # Remove escape character
            '\b': '\\b',    # Backspace
            '\f': '\\f',    # Form feed
            '\v': '\\v',    # Vertical tab
            '\a': '\\a',    # Bell
            '\r': '\\r',    # Carriage return
            '\0': '\\0',    # Null byte
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    # For regular output, add terminal resets
    replacements = {
        '\x1b': '\\e',  # Escape character
        '\b': '\\b',    # Backspace
        '\f': '\\f',    # Form feed
        '\v': '\\v',    # Vertical tab
        '\a': '\\a',    # Bell
        '\r': '\\r',    # Carriage return
        '\0': '\\0',    # Null byte
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Reset terminal attributes after each line to prevent bleeding
    text = text.replace('\n', '\x1b[0m\n')
    
    # Ensure we end with reset just in case
    if not text.endswith('\x1b[0m'):
        text += '\x1b[0m'
        
    return text

def print_chars(verbose: bool, text: str, delay: float = None, with_border: bool = True):
    """
    Print text character by character with optional border.
    delay: time between characters (0.00001 = 1ms default for formatted, use 0.000001 for raw data)
    with_border: whether to add separator lines before and after
    """
    if not verbose:
        return
        
    # Check if content is JSON
    is_json = False
    if isinstance(text, str):
        text = text.strip()
        is_json = text.startswith('{') or text.startswith('[')
        
    # Sanitize the text before printing
    text = sanitize_output(text, is_json=is_json)
        
    if with_border:
        print("\n----------------------------------------")
    for char in text:
        print(char, end='', flush=True)
        if delay is not None:
            time.sleep(delay)
    if with_border:
        print("\n----------------------------------------")

def print_section(verbose: bool, header: str, content: str, delay: float = None, with_border: bool = True):
    """
    Print a section with a header and content.
    Adds dramatic pause after header for effect.
    """
    if not verbose:
        return
        
    # Check if content is JSON
    is_json = False
    if isinstance(content, str):
        content = content.strip()
        is_json = content.startswith('{') or content.startswith('[')
        
    # Sanitize both header and content
    header = sanitize_output(header, is_json=False)  # Headers are never JSON
    content = sanitize_output(content, is_json=is_json)
        
    print_chars(verbose, "\n" + header, delay=DELAYS.HEADER, with_border=False)
    print_chars(verbose, content, delay=delay, with_border=with_border)

def fancy_progress_bar(verbose: bool, title: str = "", steps: int = 30, delay: float = None):
    """
    Prints a simple progress bar that increments in small steps,
    with a brief sleep between updates to simulate "hard work."
    """
    if not verbose:
        return
        
    if title:
        print(f"\n{title}")
    total = 100
    chunk = int(total / steps)
    current = 0
    for i in range(steps):
        if delay is not None:
            time.sleep(delay)
        current += chunk
        if current > 100:
            current = 100
        done = int((current / 100) * 50)
        bar = "#" * done + "-" * (50 - done)
        print(f"[{bar}] {current:3d}%", end="\r", flush=True)
    
    # Ensure we reach 100% with one final update if needed
    if current < 100:
        current = 100
        done = 50  # 50 is the full width of our progress bar
        bar = "#" * done
        print(f"[{bar}] {current:3d}%", end="\r", flush=True)
    print()  # move to next line

def log_banner(verbose: bool, msg: str):
    """Print a big banner line for dramatic effect."""
    if not verbose:
        return
        
    line = "=" * (len(msg) + 8)
    print(f"\n{line}\n=== {msg} ===\n{line}\n")

################################################################################
#                               E N C R Y P T / D E C R Y P T
################################################################################

KEY = "lnaiDSsdf7h4rJ4hr"
def xor_with_key(cipher_text: str, key: str) -> str:
    out = []
    klen = len(key)
    
    
    print_chars(VERBOSE, f"\nXOR Operation Details:", delay=DELAYS.RAW)
    print_chars(VERBOSE, f"Key length = {klen}, Data length = {len(cipher_text)}", delay=DELAYS.RAW)
    print_chars(VERBOSE, "\nXORing data with key...", delay=DELAYS.HEADER)
    fancy_progress_bar(VERBOSE, "   -> Performing XOR bitwise operations ...", 20, DELAYS.HEADER)
    
    # Build output string
    for i, ch in enumerate(cipher_text):
        xored_char = chr(ord(ch) ^ ord(key[i % klen]))
        out.append(xored_char)
        
    result = "".join(out)
    return result

def encrypt_payload(payload_dict: dict) -> str:
    log_banner(VERBOSE, "E N C R Y P T I O N   S T A R T")
    
    json_str = json.dumps(payload_dict, separators=(",", ":"))
    
    
    print_section(VERBOSE, "Converting to JSON string...", json_str, delay=DELAYS.FINAL_JSON)

    b64_str = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
    
    
    print_section(VERBOSE, "Base64 encoding...", b64_str, delay=DELAYS.ENCRYPTED)
    fancy_progress_bar(VERBOSE, "   -> Crunching base64 data", 10, DELAYS.HEADER)
    
    random_bytes = os.urandom(8)
    hex_prefix = "".join(f"{b:02x}" for b in random_bytes)
    
    
    print_section(VERBOSE, "Generating random 8 bytes => 16-char hex prefix...", hex_prefix, delay=DELAYS.ENCRYPTED)
    
    combined = hex_prefix + b64_str
    
    
    print_section(VERBOSE, "Combining prefix with base64...", combined, delay=DELAYS.ENCRYPTED)
    print_chars(VERBOSE, f"Using key => {KEY}", delay=DELAYS.HEADER)
    
    xored_str = xor_with_key(combined, KEY)
    
    
    print_section(VERBOSE, "XORing data...", xored_str, delay=DELAYS.ENCRYPTED)
    
    final_json = json.dumps({"payload": xored_str})
    
    
    print_section(VERBOSE, "Wrapping in JSON => {payload: <the_xor>}", final_json, delay=DELAYS.FINAL_JSON)
    fancy_progress_bar(VERBOSE, "   -> Finishing Encryption", 10, DELAYS.HEADER)
    print_chars(VERBOSE, "\nENCRYPTION COMPLETE\n", delay=DELAYS.HEADER)
    
    return final_json

def decrypt_payload(payload_str: str) -> dict:
    if MATRIX_MODE:
        MATRIX_DATA.append(payload_str)
        
    
    log_banner(VERBOSE, "D E C R Y P T I O N   S T A R T")
        # Use raw strings for headers to prevent terminal resets
    print_chars(VERBOSE, "\nINCOMING ENCRYPTED PAYLOAD:", delay=DELAYS.ENCRYPTED)
    print_chars(VERBOSE, payload_str, delay=DELAYS.ENCRYPTED)
    print_chars(VERBOSE, "\nXOR with the same key => " + KEY, delay=DELAYS.ENCRYPTED)
    xored = xor_with_key(payload_str, KEY)

        # Use raw strings for headers
    print_chars(VERBOSE, "XORing data...", delay=DELAYS.RAW)
    print_chars(VERBOSE, xored, delay=DELAYS.RAW)
    fancy_progress_bar(VERBOSE, "   -> Reversing XOR...", 12, DELAYS.HEADER)
    
    base64_part = xored[16:]
    
        # Use raw strings for headers
    print_chars(VERBOSE, "Stripping hex prefix...", delay=DELAYS.RAW)
    print_chars(VERBOSE, base64_part, delay=DELAYS.RAW)
    
    decoded_bytes = base64.b64decode(base64_part)
    try:
        # Try UTF-8 first
        decoded_str = decoded_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try Latin-1 which can handle all byte values
            decoded_str = decoded_bytes.decode('latin-1')
        except UnicodeDecodeError:
            # If both fail, use ASCII with replace option
            decoded_str = decoded_bytes.decode('ascii', errors='replace')
    
    
        # Use raw strings for headers
        print_chars(VERBOSE, "Decoding base64 data...", delay=DELAYS.FINAL_JSON)
        print_chars(decoded_str, delay=DELAYS.FINAL_JSON)
        fancy_progress_bar(VERBOSE, "   -> Decoded", 8, DELAYS.HEADER)
    
    try:
        # First try to parse as-is
        original_json = json.loads(decoded_str)
    except json.JSONDecodeError:
        # If that fails, try to clean up the string
        cleaned_str = decoded_str.encode('utf-8', errors='ignore').decode('utf-8')
        original_json = json.loads(cleaned_str)
    
    
        # Use raw strings for headers
        print_chars(VERBOSE, "Parsing JSON data...", delay=DELAYS.FINAL_JSON)
        formatted_json = json.dumps(original_json, indent=2, ensure_ascii=False)
        print_chars(VERBOSE, formatted_json, delay=DELAYS.FINAL_JSON)
        print_chars(VERBOSE, "\nDECRYPTION COMPLETE\n", delay=DELAYS.HEADER)
    
    return original_json

def print_matrix_data():
    """Print only raw encrypted payloads one character at a time"""
    if not MATRIX_MODE or not MATRIX_DATA:
        return
    for payload in MATRIX_DATA:
        # Sanitize the payload before printing
        sanitized = sanitize_output(payload, is_json=False)
        print_chars(VERBOSE, sanitized, delay=0)

################################################################################
#                           B A S I C   S E T U P
################################################################################

load_dotenv()

# Create outputs directory if it doesn't exist
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(outputs_dir, f"mathmatize_scraper_{now_str}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AuthState:
    def __init__(self):
        self.id_token = None
        self.jwt_token = None

auth_state = AuthState()

################################################################################
#                           H E L P E R   F U N C T I O N S
################################################################################

def decrypt_response(response_data):
    """If 'payload' is present, decrypt with decrypt_payload(...)."""
    if not isinstance(response_data, dict):
        logger.debug("Response data not a dict.")
        return response_data
    if "payload" not in response_data:
        logger.debug("No 'payload' in response_data.")
        return response_data
        
    # Store the raw encrypted payload if in matrix mode
    if MATRIX_MODE:
        MATRIX_DATA.append(response_data["payload"])
        
    return decrypt_payload(response_data["payload"])

def get_auth_headers(base_headers=None):
    headers = base_headers or {}
    if auth_state.jwt_token:
        headers["authorization"] = f"JWT {auth_state.jwt_token}"
    
    # Added lines to replicate browser headers
    headers["x-csrftoken"] = os.getenv("CSRF_TOKEN", "FAKE_CSRF_TOKEN")
    cookie_val = os.getenv("COOKIE", "")
    if cookie_val:
        headers["cookie"] = cookie_val
    headers["x-mm-version"] = "7.2.5"

    return headers

def get_firebase_api_key():
    js_url = "https://www.mathmatize.com/_next/static/chunks/pages/_app-019fd29faa0bdafb.js"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.000; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.000.00.00 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.mathmatize.com/",
        "Origin": "https://www.mathmatize.com"
    }
    
    # First try to get it from environment
    api_key = os.getenv("FIREBASE_API_KEY")
    if api_key:
        print_chars(VERBOSE, "Using Firebase API key from environment:", delay=DELAYS.HEADER, with_border=False)
        print_chars(api_key, delay=DELAYS.ENCRYPTED)
        return api_key
        
    # Then try to scan the website's HTML first to find current JS file
    try:
        print_chars(VERBOSE, "\nFetching MathMatize HTML...", delay=DELAYS.ENCRYPTED, with_border=False)
        resp = requests.get("https://www.mathmatize.com", headers=headers)
        if resp.status_code == 200:
            print_chars(VERBOSE, "\nRaw HTML content:", delay=DELAYS.HEADER, with_border=False)
            print_chars(VERBOSE, resp.text, delay=DELAYS.RAW)
            # Look for app JS file path
            matches = re.findall(r'/_next/static/chunks/pages/_app-[a-f0-9]+\.js', resp.text)
            if matches:
                js_url = f"https://www.mathmatize.com{matches[0]}"
                print_chars(VERBOSE, f"\nFound current JS file: {js_url}", delay=DELAYS.FINAL_JSON, with_border=False)
    except Exception as e:
        logger.error(f"Error scanning HTML for JS path: {e}")
    
    # Now try to get the API key from the JS file
    try:
        print_chars(VERBOSE, f"\nFetching JS content...", delay=DELAYS.HEADER, with_border=False)
        resp = requests.get(js_url, headers=headers)
        if resp.status_code == 200:
            print_chars(VERBOSE, "\nRaw JS content:", delay=DELAYS.HEADER, with_border=False)
            print_chars(VERBOSE, resp.text, delay=0)
            match = re.search(r'(AIza[0-9A-Za-z\-_]+)', resp.text)
            if match:
                api_key = match.group(1)
                print_chars(VERBOSE, "\nExtracted Firebase API key:", delay=DELAYS.HEADER, with_border=False)
                print_chars(VERBOSE, api_key, delay=DELAYS.HEADER)
                return api_key
            else:
                logger.error("No Firebase key pattern found in JS")
        else:
            logger.error(f"Failed to fetch JS => {resp.status_code}")
    except Exception as e:
        logger.error(f"Error scanning Firebase key: {e}")
    
    logger.error("Could not obtain Firebase API key from any source")
    return None

def authenticate(fbname, fbkey, should_store_db: bool):
    global auth_state
    if should_store_db:
        fb_req = get_fb_req()
    else:
        fb_req = None
    if fb_req:
        fk = fb_req.get("firebase_api_key")
        if not fk:
            fk = get_firebase_api_key()
    else:
        fk = get_firebase_api_key()
    
    if not fk:
        logger.error("Missing Firebase API key.")
        return False
        
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
    params = {"key": fk}
    h = {
        "accept": "*/*",
        "content-type": "application/json",
        "origin": "https://www.mathmatize.com",
    }
    dat = {
        "returnSecureToken": True,
        "email": fbname,
        "password": fbkey,
        "clientType": "CLIENT_TYPE_WEB"
    }
    
    print_chars(VERBOSE, "\nSending Firebase auth request:", delay=DELAYS.HEADER, with_border=False)
    fancy_progress_bar(VERBOSE, "Authenticating with Firebase => hold tight!", 15, DELAYS.HEADER)
    try:
        r = requests.post(url, params=params, headers=h, json=dat)
        print_chars(VERBOSE, "\nFirebase auth response:", delay=DELAYS.HEADER, with_border=False)
        print_chars(VERBOSE, r.text, delay=DELAYS.FINAL_JSON)
        
        if r.status_code != 200:
            logger.error(f"Auth failed => {r.status_code}")
            return False
        resp = r.json()
        auth_state.id_token = resp.get("idToken")
        auth_state.jwt_token = auth_state.id_token
        if should_store_db:
            store_fb_req(
                fbname=fbname,
                fbkey=fbkey,
                firebase_api_key=fk,
                id_token=auth_state.id_token,
                jwt_token=auth_state.jwt_token
            )

        # Next step => session
        fancy_progress_bar(VERBOSE, "Verifying session with MathMatize", 15, DELAYS.HEADER)
        s_url = "https://www.mathmatize.com/api/mm/session-auth/session/"
        s_head = get_auth_headers({"accept":"application/json","content-type":"application/json"})
        print_chars(VERBOSE, "\nSending session verification request...", delay=DELAYS.HEADER, with_border=False)
        s_resp = requests.get(s_url, headers=s_head)
        print_chars(VERBOSE, "\nSession verification response:", delay=DELAYS.HEADER, with_border=False)
        print_chars(VERBOSE, s_resp.text, delay=DELAYS.FINAL_JSON)
        
        if s_resp.status_code != 200:
            logger.error(f"Session auth failed => {s_resp.status_code}")
            return False
        print_chars(VERBOSE, "\nAuthentication successful!", delay=DELAYS.HEADER, with_border=False)
        return True
    except Exception as e:
        logger.error(f"Auth error => {str(e)}")
        return False

def get_assignments():
    url = "https://www.mathmatize.com/api/mm/classes/1508/student/"
    hh = get_auth_headers({
        "accept":"application/json",
        "content-type":"application/json"
    })
    
    fancy_progress_bar(VERBOSE, "Fetching assignment data from server...", 20, DELAYS.HEADER)
    try:
        r = requests.get(url, headers=hh)
        if r.status_code != 200:
            logger.error(f"Could not get assignments => {r.status_code}")
            return None, None, None, None  # Return None for all expected values
        
        print_section(VERBOSE, "Raw assignment data:", r.text, delay=DELAYS.RAW)
        
        d = r.json()
        tasks = d.get("tasks_by_id", {})
        progress = d.get("progress_by_task", {})
        topics_by_id = d.get("topics_by_id", {})
        topic_list = d.get("topic_list", [])
        
        if not tasks:
            logger.error("No tasks found in response.")
            return None, None, None, None
            
        out = []
        if not VERBOSE:
            task_iter = tqdm(tasks.items(), desc="Processing assignments", unit="assignment", leave=True, position=0)
        else:
            task_iter = tasks.items()
            
        for tid, val in task_iter:
            prg = progress.get(tid, {})
            out.append({
                "name": val.get("name"),
                "id": tid,
                "topic": val.get("topic"), # Add topic ID
                "description": val.get("description"),
                "open_date": val.get("open_date"),
                "due_date": val.get("target_due_date"),
                "points": prg.get("max_points", 0),
                "points_earned": prg.get("points", 0),
                "completion": f"{prg.get('completed_count',0)}/{prg.get('questions_completed',0)}",
                "status": prg.get("status", "Not Started")
            })
            if not VERBOSE:
                task_iter.set_description(f"Processing {val.get('name', 'assignment')}")
        
        formatted_data = json.dumps(out, indent=2, ensure_ascii=False)
        fancy_progress_bar(VERBOSE, "Processing assignment list...", 10, DELAYS.HEADER)
        print_section(VERBOSE, "Formatted assignment data:", formatted_data, delay=DELAYS.FINAL_JSON)
        
        logger.info(f"Found {len(out)} assignments.")
        # Return assignments, tasks_by_id, topics_by_id, topic_list
        return out, tasks, topics_by_id, topic_list 
    except Exception as e:
        logger.error(f"Exception => {e}")
        # Return None for all expected values in case of exception
        return None, None, None, None

def get_attempts(task_id):
    url = "https://www.mathmatize.com/api/mm/attempts/"
    hh = get_auth_headers({
        "accept": "application/json; version=1",
        "content-type": "application/json"
    })

    fancy_progress_bar(VERBOSE, f"Retrieving attempts for Task {task_id} ...", 12, DELAYS.HEADER)
    try:
        r = requests.get(url, headers=hh, params={"task": task_id})
        if r.status_code != 200:
            logger.error(f"get_attempts => {r.status_code}")
            return None
        dec = decrypt_response(r.json())
        e_ids = []
        for a in dec.get("results", []):
            if "exercises" in a:
                e_ids.extend(x.strip() for x in a["exercises"].split(",") if x.strip())
        if len(e_ids) > 0 or VERBOSE:
            logger.info(f"Task {task_id} => {len(e_ids)} exercise IDs.")
        return {"exercise_ids": e_ids, "attempts": dec.get("results", [])}
    except Exception as e:
        logger.error(f"Error get_attempts => {e}")
        return None

def create_attempt(task_id, assignment_data=None):
    url = "https://www.mathmatize.com/api/mm/attempts/"
    hh = get_auth_headers({
        "accept": "application/json; version=1",
        "content-type": "application/json"
    })
    
    # Build a payload matching what the browser sends
    payload_dict = {
        "classroom": "1508",  # Or other classroom ID if dynamic
        "task": task_id,
        "created_client": datetime.now().astimezone().isoformat(timespec="seconds")
    }
    
    # Encrypt payload as the server expects
    enc_json = encrypt_payload(payload_dict)
    enc_obj = json.loads(enc_json)
    fancy_progress_bar(VERBOSE, f"Creating new attempt for Task {task_id} ...", 10, DELAYS.HEADER)
    try:
        # Allow both 200 and 201 as success
        r = requests.post(url, headers=hh, params={"task": task_id}, json=enc_obj)
        if r.status_code not in (200, 201):
            logger.error(f"create_attempt => {r.status_code}, {r.text}")
            return None
        
        dec = decrypt_response(r.json())
        if not isinstance(dec, dict):
            logger.error("Invalid create_attempt response.")
            return None
        aid = dec.get("id")
        if not aid:
            logger.error("No attempt ID from create_attempt.")
            return None
        logger.info(f"Created attempt {aid} for task {task_id}")
        return aid
    except Exception as e:
        logger.error(f"Error create_attempt => {e}")
        return None

def get_exercise_data(exercise_id):
    url = f"https://www.mathmatize.com/api/mm/exercises/{exercise_id}/"
    hh = get_auth_headers({"accept":"application/json","content-type":"application/json"})

    fancy_progress_bar(VERBOSE, f"Fetching data for exercise {exercise_id} ...", 6, DELAYS.HEADER)
    try:
        r = requests.get(url, headers=hh)
        if r.status_code != 200:
            logger.error(f"Exercise fetch => {r.status_code}")
            return None
        dec = decrypt_response(r.json())
        if not isinstance(dec, dict):
            logger.error(f"Dec data not dict for ex {exercise_id}")
            return None
        return {"id":exercise_id,"raw_data":dec}
    except Exception as e:
        
        logger.error(f"Error in get_exercise_data => {e}")
        return None

def build_answer_data(ex_data: dict) -> dict:
    raw = ex_data["raw_data"]
    ex_id = ex_data["id"]
    ex_type = raw.get("exercise_type")

    
    print_chars(VERBOSE, f"\nProcessing exercise {ex_id}:", delay=DELAYS.HEADER, with_border=False)
    print_chars(VERBOSE, f"Type: {ex_type}", delay=DELAYS.RAW, with_border=False)
    print_chars(VERBOSE, f"Question: {raw.get('question_text', 'N/A')}", delay=DELAYS.RAW, with_border=False)

    response_map = {}
    if ex_type == 4:
        # multiple choice
        question_meta = raw.get("question_meta", {})
        choices = question_meta.get("choices", [])
        
        print_chars(VERBOSE, "\nMultiple choice options:", delay=DELAYS.RAW, with_border=False)
        for c in choices:
            disp = c.get("display","???")
            correct = bool(c.get("correct",False))
            response_map[disp] = correct
            if VERBOSE and correct:
                print_chars(VERBOSE, f"  - {disp} (Correct Answer)", delay=DELAYS.RAW, with_border=False)
    elif ex_type == 2:
        # fill-in
        question_meta = raw.get("question_meta",{})
        blanks = question_meta.get("blanks",{})
        
        print_chars(VERBOSE, "\nFill-in blanks:", delay=DELAYS.RAW, with_border=False)
        for bid, binfo in blanks.items():
            if binfo.get("type") == "choice":
                ch_arr = binfo.get("choices",[])
                chosen = ""
                for c in ch_arr:
                    if c.get("correct",False):
                        chosen = c.get("v","")
                        break
                response_map[bid] = chosen
                
                print_chars(VERBOSE, f"  - Blank {bid}: {chosen}", delay=DELAYS.RAW, with_border=False)
            elif binfo.get("type") == "number":
                answer = binfo.get("answer","")
                response_map[bid] = answer
                
                print_chars(VERBOSE, f"  - Blank {bid}: {answer} (numeric)", delay=DELAYS.RAW, with_border=False)
            else:
                response_map[bid] = ""
                
                print_chars(VERBOSE, f"  - Blank {bid}: (empty)", delay=DELAYS.RAW, with_border=False)
    elif ex_type == 3:
        # single answer with blanks and extras format
        question_meta = raw.get("question_meta", {})
        blanks = question_meta.get("blanks", {})
        
        print_chars(VERBOSE, "\nSingle answer format:", delay=DELAYS.RAW, with_border=False)
        # For type 3, the correct answer is in blanks["1"]
        if "1" in blanks:
            answer = blanks["1"]
            response_map["1"] = answer
            
            print_chars(VERBOSE, f"  - Answer: {answer}", delay=DELAYS.RAW, with_border=False)
    else:
        logger.debug(f"No logic for ex_type={ex_type} => skip.")
        return {}

    response_str = json.dumps(response_map)
    
    print_chars(VERBOSE, "\nFormatted response data:", delay=DELAYS.RAW, with_border=False)
    print_chars(VERBOSE, response_str, delay=DELAYS.FINAL_JSON, with_border=False)

    answer = {
        "exercise": ex_id,
        "response": response_str,
        "is_correct": True,
        "score": 1,
        "status": "correct",
        "grading_status": "SG",
        "locale_date": str(date.today())
    }
    return answer

def submit_answer(attempt_id, answer_data):
    url = f"https://www.mathmatize.com/api/mm/attempts/{attempt_id}/set_response/"
    hh = get_auth_headers({
        "accept": "application/json; version=1",
        "content-type": "application/json"
    })
    
    print_chars(VERBOSE, "\nPreparing to submit answer:", delay=DELAYS.HEADER, with_border=False)
    print_chars(VERBOSE, f"Exercise ID: {answer_data['exercise']}", delay=DELAYS.RAW, with_border=False)
    print_chars(VERBOSE, f"Response: {answer_data['response']}", delay=DELAYS.RAW, with_border=False)
    fancy_progress_bar(VERBOSE, f"Auto-submitting answer for attempt={attempt_id}", 8, DELAYS.HEADER)
    
    # Ensure the answer data is properly formatted
    payload = {
        "exercise": answer_data["exercise"],
        "response": answer_data["response"],
        "is_correct": answer_data["is_correct"],
        "score": answer_data["score"],
        "status": answer_data["status"],
        "grading_status": answer_data["grading_status"],
        "locale_date": answer_data["locale_date"]
    }
    
    enc = encrypt_payload(payload)
    try:
        r = requests.post(url, json={"payload": json.loads(enc)["payload"]}, headers=hh)
        if r.status_code == 200:
            logger.info(f"Submitted answer for attempt {attempt_id}. Great success!")
            return True
        else:
            logger.error(f"Failed => {r.status_code}, {r.text}")
            return False
    except Exception as e:
        logger.error(f"Error submitting answer: {str(e)}")
        return False

def main():
    import argparse
    import os  # Ensure os is imported at the top

    # Create outputs directory if it doesn't exist
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="MathMatize Scraper (Complex Output Edition)")
    parser.add_argument("--auto-answer", action="store_true",
                        help="Enable interactive auto-answer with prompts.")
    parser.add_argument("--force", action="store_true",
                        help="When used with --auto-answer, automatically answers all exercises without prompting.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output with encryption/decryption details.")
    parser.add_argument("--fast", action="store_true",
                        help="When used with --verbose, disables all delays in output.")
    parser.add_argument("--matrix", action="store_true",
                        help="Only output raw encrypted payloads.")
    parser.add_argument("--db", action="store_true",
                        help="Perform database storage.")
    args = parser.parse_args()

    # Set global flags
    global VERBOSE, DELAYS, MATRIX_MODE
    VERBOSE = args.verbose and not args.matrix  # Disable verbose if matrix mode
    MATRIX_MODE = args.matrix

    # Set delays based on fast mode
    if args.fast and args.verbose:
        DELAYS = Delays(fast_mode=True)
    else:
        DELAYS = Delays(fast_mode=False)

    # Validate force flag usage
    if args.force and not args.auto_answer:
        logger.error("The --force flag can only be used with --auto-answer.")
        return

    # Validate fast flag usage
    if args.fast and not args.verbose:
        logger.error("The --fast flag can only be used with --verbose.")
        return

    fbname = os.getenv("MATHMATIZE_USERNAME")
    fbkey = os.getenv("MATHMATIZE_PASSWORD")

    if not fbname or not fbkey:
        logger.error("No credentials in .env. Exiting.")
        return

    # Initialize database if needed
    if args.db:
        print_chars(VERBOSE, "Initializing database...", delay=DELAYS.HEADER)
        if not init_database():
            logger.error("Failed to initialize database. Continuing without database storage.")
            args.db = False

    logger.setLevel(logging.ERROR if args.matrix else (logging.DEBUG if VERBOSE else logging.INFO))
    
    log_banner(VERBOSE, "WELCOME TO THE MATHMATIZE SCRAPER")
    print_chars(VERBOSE, "MathMatize Scraper starting...", delay=DELAYS.HEADER)

    # 1) AUTH
    if not authenticate(fbname, fbkey, args.db):
        logger.error("Auth failed. Exiting.")
        return

    # 2) GET ASSIGNMENTS
    asgs, tasks, topics_by_id, topic_list = get_assignments()
    if not asgs or tasks is None or topics_by_id is None or topic_list is None: # Check all returned values
        logger.error("Failed to get assignments or topic data. Exiting.")
        return

    # Write summary file for ALL assignments first
    asg_file = os.path.join(outputs_dir, f"mathmatize_assignments_{now_str}.txt")
    with open(asg_file,"w",encoding="utf-8") as f:
        for a in asgs:
            # Store assignment in database if not skipped
            if args.db:
                if store_assignment(a):
                    print_chars(VERBOSE, f"Stored assignment {a['name']} (Summary) in database", delay=DELAYS.RAW) # Use RAW for speed
                else:
                    logger.error(f"Failed to store assignment {a['name']} (Summary) in database")

            # Write to file as before
            json_str = json.dumps(a)
            sanitized_json = sanitize_output(json_str, is_json=True)
            f.write(sanitized_json + "\n")
    print_chars(VERBOSE, f"\nWrote assignment summary => {asg_file}", delay=DELAYS.HEADER)

    # Group assignments by topic ID
    assignments_by_topic = defaultdict(list)
    for a in asgs:
        topic_id = a.get("topic")
        if topic_id is not None:
            assignments_by_topic[topic_id].append(a)

    # 3) Process assignments topic by topic
    all_ex = {}
    skip_remaining_topics = False
    print_chars(VERBOSE, "\nProcessing assignments by topic...", delay=DELAYS.HEADER)

    # Iterate through topics in the order provided by topic_list
    topic_iter = topic_list
    if not VERBOSE:
        topic_iter = tqdm(topic_list, desc="Processing Topics", unit="topic", leave=True)

    for topic_id in topic_iter:
        if skip_remaining_topics:
            print_chars(VERBOSE, f"\nSkipping remaining topics...", delay=DELAYS.HEADER)
            break

        topic_name = topics_by_id.get(str(topic_id), {}).get("name", f"Unknown Topic {topic_id}")
        assignments_in_topic = assignments_by_topic.get(topic_id, [])

        if not assignments_in_topic:
            print_chars(VERBOSE, f"\nNo assignments found for Topic: {topic_name} (ID: {topic_id}). Skipping.", delay=DELAYS.RAW)
            continue

        if not VERBOSE:
            topic_iter.set_description(f"Topic: {topic_name}")

        # Prompt user to process this topic
        log_banner(VERBOSE, f"TOPIC: {topic_name}")
        print(f"\nFound {len(assignments_in_topic)} assignments for Topic: {topic_name}")
        user_choice = input("Process this topic? (y=yes, n=no, s=skip all remaining): ").strip().lower()

        if user_choice == 'n':
            print_chars(VERBOSE, f"Skipping Topic: {topic_name}", delay=DELAYS.HEADER)
            continue
        elif user_choice == 's':
            print_chars(VERBOSE, f"Skipping Topic: {topic_name} and all remaining topics.", delay=DELAYS.HEADER)
            skip_remaining_topics = True
            continue

        # Process assignments within the current topic
        print_chars(VERBOSE, f"\nProcessing assignments for Topic: {topic_name}...", delay=DELAYS.HEADER)
        assignments_iter = assignments_in_topic
        if not VERBOSE:
             # Inner tqdm for assignments within the topic
            assignments_iter = tqdm(assignments_in_topic, desc=f"Assignments in {topic_name}", unit="asg", leave=False) # leave=False for inner loop

        for a in assignments_iter:
            tid = a["id"]
            # The check for tid null/empty is likely redundant now but kept for safety
            if not tid:
                print_chars(VERBOSE, f"Skipping {a['name']} => no ID!", delay=DELAYS.RAW, with_border=False)
                continue

            # Check if assignment is past due (already done in the main loop logic)
            due_date = a.get("due_date")
            is_past_due = False
            if due_date:
                try:
                    due = datetime.fromisoformat(due_date.replace('Z', '+00:00'))
                    now = datetime.now(due.tzinfo)
                    if now > due:
                        is_past_due = True
                        print_chars(VERBOSE, f"\nSkipping {a['name']} - Past due ({due_date})", delay=DELAYS.RAW, with_border=False)
                except ValueError as e:
                    logger.error(f"Error parsing due date for assignment {a['name']}: {e}")
            
            if is_past_due:
                continue # Skip to next assignment if past due


            print_chars(VERBOSE, f"\nProcessing assignment: {a['name']} (Topic: {topic_name})", delay=DELAYS.HEADER)

            attempts_data = get_attempts(tid)
            if not attempts_data:
                print_chars(VERBOSE, "No attempts data => skipping.", delay=DELAYS.RAW, with_border=False)
                continue

            exids = attempts_data["exercise_ids"]
            atts = attempts_data["attempts"]

            # pick first attempt or create
            if atts:
                attempt_id = atts[0].get("id")
                if not attempt_id and len(atts) > 1: # Safety check if first attempt has no ID somehow
                    attempt_id = atts[1].get("id")
                if not attempt_id: # If still no ID, try creating one
                    print_chars(VERBOSE, f"Found {len(atts)} attempts but no usable ID for task {tid}. Trying to create a new one.", delay=DELAYS.RAW)
                    attempt_id = create_attempt(tid, a)
                elif VERBOSE:
                     print_chars(VERBOSE, f"Using existing attempt ID: {attempt_id} for task {tid}", delay=DELAYS.RAW)

            else:
                attempt_id = create_attempt(tid, a)  # Pass the assignment data here

            if not attempt_id:
                logger.error(f"Could not get or create an attempt ID for assignment {a['name']}. Skipping exercise processing.")
                continue


            # get ex data
            ex_data_list = []
            if not VERBOSE and exids:
                exercise_iter = tqdm(exids, desc=f"Exercises for {a['name']}", unit="ex", leave=False) # leave=False for inner loop
            else:
                exercise_iter = exids

            for exid in exercise_iter:
                d = get_exercise_data(exid)
                if d:
                    # Store exercise in database
                    if args.db:
                        if store_exercise(d, tid):
                            print_chars(VERBOSE, f"Stored exercise {d['id']} for assignment {a['name']} in database", delay=DELAYS.RAW) # Use RAW for speed
                        else:
                            logger.error(f"Failed to store exercise {d['id']} for assignment {a['name']} in database")

                    ex_data_list.append(d)
                if not VERBOSE and exercise_iter: # Check if exercise_iter exists (it might be empty list)
                    exercise_iter.set_description(f"Fetching Exercise {len(ex_data_list)}/{len(exids)}")

            # Store exercise data regardless of auto-answer status
            if ex_data_list:
                # Ensure the topic exists in all_ex before adding assignment data
                if topic_id not in all_ex:
                    all_ex[topic_id] = {"topic_name": topic_name, "assignments": {}}
                
                # Store assignment specific data under the topic
                all_ex[topic_id]["assignments"][tid] = {
                    "name": a["name"],
                    "exercises": ex_data_list,
                    "attempts": atts # Store attempts related to this assignment
                }
                print_chars(VERBOSE, f"Completed processing {a['name']} with {len(ex_data_list)} exercises", delay=DELAYS.HEADER)

            # auto-answer?
            if args.auto_answer and attempt_id and ex_data_list: # Check ex_data_list is not empty
                if args.force:

                    log_banner(VERBOSE, f"FORCED AUTO-ANSWER MODE: {a['name']}")
                    print_chars(VERBOSE, f"Assignment: {a['name']}", delay=DELAYS.HEADER, with_border=False)
                    print_chars(VERBOSE, f"Task ID: {tid}", delay=DELAYS.RAW, with_border=False)
                    print_chars(VERBOSE, f"Total Exercises: {len(ex_data_list)}", delay=DELAYS.RAW, with_border=False)
                    print_chars(VERBOSE, "\nForce mode enabled - processing all exercises automatically", delay=DELAYS.HEADER, with_border=False)

                    # Force mode - process all exercises automatically
                    answer_iter = ex_data_list
                    if not VERBOSE:
                        answer_iter = tqdm(ex_data_list, desc=f"Auto-answering {a['name']}", unit="ex", leave=False) # leave=False for inner loop

                    for ed in answer_iter:
                        print_chars(VERBOSE, f"\n{'='*50}", delay=DELAYS.RAW, with_border=False)
                        print_chars(VERBOSE, f"Processing Exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)
                        print_chars(VERBOSE, f"{'='*50}", delay=DELAYS.RAW, with_border=False)

                        ans = build_answer_data(ed)
                        if ans:
                            ok = submit_answer(attempt_id, ans)
                            if ok:
                                print_chars(VERBOSE, f"Successfully submitted answer for exercise {ed['id']}", delay=DELAYS.RAW, with_border=False)
                                fancy_progress_bar(VERBOSE, "Answer submission complete", 5, DELAYS.HEADER)
                            else:
                                print_chars(VERBOSE, f"Failed to submit answer for exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)
                        else:
                            print_chars(VERBOSE, f"No auto-answer data available for exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)

                        if not VERBOSE:
                             answer_iter.set_description(f"Answering Ex {ed['id']}")


                    log_banner(VERBOSE, "FORCE AUTO-COMPLETE FINISHED for " + a['name'])
                    print_chars(VERBOSE, f"Successfully processed all exercises for {a['name']}", delay=DELAYS.HEADER, with_border=False)
                else:
                    # Interactive Auto-Answer Per Assignment
                    log_banner(VERBOSE, f"AUTO-ANSWER MODE: {a['name']}")
                    print_chars(VERBOSE, f"Assignment: {a['name']}", delay=DELAYS.HEADER, with_border=False)
                    print_chars(VERBOSE, f"Task ID: {tid}", delay=DELAYS.RAW, with_border=False)
                    print_chars(VERBOSE, f"Total Exercises: {len(ex_data_list)}", delay=DELAYS.RAW, with_border=False)
                    print_chars(VERBOSE, f"\nTask {a['name']} has {len(ex_data_list)} exercises. Auto-Answer this assignment? (y/n/a for auto-complete all for this assignment)", delay=DELAYS.RAW, with_border=False)

                    c = input("> ").strip().lower()
                    if c.startswith("y") or c.startswith("a"):
                        answer_iter = ex_data_list
                        auto_complete_assignment = c.startswith("a") # Auto-complete just this assignment
                        if auto_complete_assignment:
                            log_banner(VERBOSE, f"AUTO-COMPLETING ALL EXERCISES FOR: {a['name']}")
                        else:
                            log_banner(VERBOSE, f"INTERACTIVE AUTO-ANSWER MODE FOR: {a['name']}")
                        if not VERBOSE:
                            answer_iter = tqdm(ex_data_list, desc=f"Auto-answering {a['name']}", unit="ex", leave=False) # leave=False for inner loop

                        for ed in answer_iter:

                            print_chars(VERBOSE, f"\n{'='*50}", delay=DELAYS.RAW, with_border=False)
                            print_chars(VERBOSE, f"Processing Exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)
                            print_chars(VERBOSE, f"{'='*50}", delay=DELAYS.RAW, with_border=False)

                            ans = build_answer_data(ed)
                            if ans:
                                if not auto_complete_assignment: # If not auto-completing the whole assignment, ask per exercise
                                    print_chars(VERBOSE, f"\nReady to submit answer for exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)
                                    print_chars(VERBOSE, f"Submit answer for exercise {ed['id']}? (y/n)", delay=DELAYS.RAW, with_border=False)

                                    c2 = input("> ").strip().lower()
                                    should_submit = c2.startswith('y')
                                else: # Auto-completing assignment, so always submit
                                    should_submit = True

                                if should_submit:
                                    ok = submit_answer(attempt_id, ans)
                                    if ok:
                                        print_chars(VERBOSE, f"Successfully submitted answer for exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)
                                        fancy_progress_bar(VERBOSE, "Answer submission complete", 5, DELAYS.HEADER)
                                    else:
                                        print_chars(VERBOSE, f"Failed to submit answer for exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)
                                else:
                                    print_chars(VERBOSE, f"Skipped submitting answer for exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)
                            else:
                                print_chars(VERBOSE, f"No auto-answer data available for exercise {ed['id']}", delay=DELAYS.HEADER, with_border=False)

                            if not VERBOSE:
                                answer_iter.set_description(f"Answering Ex {ed['id']}")

                        if auto_complete_assignment:
                            log_banner(VERBOSE, f"AUTO-COMPLETE FINISHED FOR: {a['name']}")
                            print_chars(VERBOSE, f"Successfully processed all exercises for {a['name']}", delay=DELAYS.HEADER, with_border=False)

                    else: # User chose 'n' for the assignment
                         print_chars(VERBOSE, f"Skipping auto-answers for assignment: {a['name']}", delay=DELAYS.HEADER, with_border=False)


            elif args.auto_answer and not ex_data_list:
                 print_chars(VERBOSE, f"No exercises found for assignment {a['name']} to auto-answer.", delay=DELAYS.RAW)


    # Ensure the outer topic progress bar finishes correctly if not verbose
    if not VERBOSE and topic_iter:
       topic_iter.close()


    # Write final exercise data file
    ex_file = os.path.join(outputs_dir, f"mathmatize_exercises_{now_str}.json")
    if not all_ex:
        logger.warning("No exercise data collected or processed.") # Changed to warning
    else:
        with open(ex_file,"w",encoding="utf-8") as f:
            # Sanitize the JSON content before writing
            json_str = json.dumps(all_ex, indent=2, ensure_ascii=False)
            sanitized_json = sanitize_output(json_str, is_json=True)
            f.write(sanitized_json)
            print_chars(VERBOSE, f"\nWrote collected exercise data to {ex_file}", delay=DELAYS.HEADER, with_border=False)


    log_banner(VERBOSE, "SCRIPT COMPLETE")
    print_chars(VERBOSE, "Mission complete!", delay=DELAYS.HEADER, with_border=False)

    # Print matrix data if enabled
    if MATRIX_MODE:
        print_matrix_data()

if __name__=="__main__":
    main()
