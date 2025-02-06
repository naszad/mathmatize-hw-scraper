#!/usr/bin/env python3
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from bs4 import BeautifulSoup
import re
import json
import zlib
import logging
import sys
import time
import base64

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
log_filename = f"mathmatize_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def xor_strings(data, key):
    """XOR each character of data with repeating key."""
    return ''.join(chr(ord(d) ^ ord(key[i % len(key)])) 
                  for i, d in enumerate(data))

def decrypt_response(response_data):
    """Decrypt MathMatize response data."""
    if not isinstance(response_data, dict):
        logger.debug("Response data is not a dictionary")
        return response_data
        
    if 'payload' not in response_data:
        logger.debug("No payload found in response data")
        logger.debug(f"Response keys: {list(response_data.keys())}")
        return response_data
        
    try:
        # Constants from JavaScript
        T = "lnaiDSsdf7h4rJ4hr"  # Encryption key
        j = 16  # Length of random prefix (2 * 8)
        
        payload = response_data['payload']
        logger.debug(f"Raw payload (first 100 chars): {payload[:100]}")
        
        # 1. XOR decrypt with key
        decrypted = xor_strings(payload, T)
        logger.debug(f"After XOR (first 100 chars): {decrypted[:100]}")
        
        # 2. Remove random prefix
        base64_data = decrypted[j:]
        logger.debug(f"After prefix removal (first 100 chars): {base64_data[:100]}")
        
        # 3. Base64 decode
        try:
            decoded_bytes = base64.b64decode(base64_data)
            logger.debug(f"Successfully decoded base64 data")
        except Exception as e:
            logger.error(f"Base64 decode failed: {str(e)}")
            return response_data
        
        # 4. Convert to UTF-8 string
        try:
            decoded_str = decoded_bytes.decode('utf-8')
            logger.debug(f"Successfully decoded UTF-8 string")
        except Exception as e:
            logger.error(f"UTF-8 decode failed: {str(e)}")
            return response_data
        
        # 5. Parse JSON
        try:
            result = json.loads(decoded_str)
            logger.debug(f"Successfully parsed JSON")
            logger.debug(f"Decrypted data keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
            return result
        except Exception as e:
            logger.error(f"JSON parse failed: {str(e)}")
            return response_data
        
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        logger.debug(f"Full response data: {json.dumps(response_data)}")
        return response_data

def get_auth_headers(base_headers=None):
    """Get headers with authentication token."""
    headers = base_headers or {}
    if auth_state.jwt_token:
        headers["authorization"] = f"JWT {auth_state.jwt_token}"
    return headers

def get_firebase_api_key():
    """Dynamically scan and return the Firebase API key from the MathMatize JavaScript file."""
    js_url = "https://www.mathmatize.com/_next/static/chunks/pages/_app-019fd29faa0bdafb.js"  # Updated URL
    try:
        resp = requests.get(js_url)
        if (resp.status_code == 200):
            # Regex to match Firebase API keys (typically starting with "AIza")
            match = re.search(r'(AIza[0-9A-Za-z\-_]+)', resp.text)
            if match:
                return match.group(1)
            else:
                logger.error("Firebase API key not found in JS file")
        else:
            logger.error(f"Failed to fetch JS file for Firebase key: {resp.status_code}")
    except Exception as e:
        logger.error(f"Error scanning Firebase API key: {str(e)}")
    # Fallback to environment variable if dynamic scan fails
    return os.getenv('FIREBASE_API_KEY')

def authenticate(username, password):
    """Authenticate with MathMatize using Firebase."""
    global auth_state
    
    # Dynamically get Firebase API key
    firebase_key = get_firebase_api_key()
    if not firebase_key:
        logger.error("Missing Firebase API key")
        return False

    # Firebase Auth endpoint
    url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
    params = {"key": firebase_key}  # Use dynamic key instead of hardcoded value
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "origin": "https://www.mathmatize.com",
        "x-client-version": "Chrome/JsCore/11.1.0/FirebaseCore-web"
    }
    json_data = {
        "returnSecureToken": True,
        "email": username,
        "password": password,
        "clientType": "CLIENT_TYPE_WEB"
    }
    
    try:
        response = requests.post(url, params=params, headers=headers, json=json_data)
        if response.status_code != 200:
            logger.error(f"Authentication failed: {response.status_code}")
            return False
            
        auth_data = response.json()
        auth_state.id_token = auth_data.get('idToken')
        auth_state.jwt_token = auth_state.id_token
        
        # Initialize session auth
        session_auth_url = "https://www.mathmatize.com/api/mm/session-auth/session/"
        session_headers = get_auth_headers({
            "accept": "application/json",
            "content-type": "application/json",
            "x-mm-version": "7.2.2"
        })
        
        session_response = requests.get(session_auth_url, headers=session_headers)
        if session_response.status_code != 200:
            logger.error(f"Session auth failed: {session_response.status_code}")
            return False
            
        logger.info("Authentication successful")
        return True
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return False

def get_assignments():
    """Get list of assignments from the student API."""
    url = "https://www.mathmatize.com/api/mm/classes/1508/student/"
    headers = get_auth_headers({
        "accept": "application/json",
        "content-type": "application/json",
        "x-mm-version": "7.2.2"
    })
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to get assignments: {response.status_code}")
            return None
            
        data = response.json()
        logger.debug(f"Raw assignments response: {json.dumps(data)[:500]}...")
        
        # Note: The assignments response is not encrypted
        tasks_by_id = data.get('tasks_by_id', {})
        progress_by_task = data.get('progress_by_task', {})  # Changed from progress_by_id
        
        if not tasks_by_id:
            logger.error("No tasks found in response data")
            logger.debug(f"Response keys: {list(data.keys())}")
            return None
            
        assignments = []
        for task_id, task in tasks_by_id.items():
            progress = progress_by_task.get(task_id, {})  # Changed from progress_by_id
            assignment = {
                'name': task.get('name'),
                'id': task_id,
                'description': task.get('description'),
                'open_date': task.get('open_date'),
                'due_date': task.get('target_due_date'),
                'points': progress.get('max_points', 0),
                'points_earned': progress.get('points', 0),
                'completion': f"{progress.get('completed_count', 0)}/{progress.get('questions_completed', 0)}",
                'status': progress.get('status', 'Not Started')
            }
            assignments.append(assignment)
            logger.debug(f"Added assignment: {json.dumps(assignment)}")
            
        logger.info(f"Found {len(assignments)} assignments")
        return assignments
        
    except Exception as e:
        logger.error(f"Error getting assignments: {str(e)}")
        logger.exception("Full traceback:")
        return None

def get_attempts(task_id):
    """Get attempts for a task with decryption."""
    url = f"https://www.mathmatize.com/api/mm/attempts/"
    params = {"task": task_id}
    headers = get_auth_headers({
        "accept": "application/json; version=1",
        "content-type": "application/json",
        "x-mm-version": "7.2.2"
    })
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to get attempts for task {task_id}: {response.status_code}")
            return None
        
        data = response.json()
        logger.debug(f"Raw attempts data for task {task_id}: {json.dumps(data)[:500]}...")
        
        decrypted_data = decrypt_response(data)
        logger.debug(f"Decrypted attempts data for task {task_id}: {json.dumps(decrypted_data)[:500]}...")
        
        # Extract exercise IDs from the exercises field
        exercise_ids = []
        for attempt in decrypted_data.get('results', []):
            if 'exercises' in attempt:
                # Split the comma-separated exercise IDs
                ids = attempt['exercises'].split(',')
                exercise_ids.extend([ex_id.strip() for ex_id in ids if ex_id.strip()])
                
        logger.info(f"Found {len(exercise_ids)} exercise IDs in attempts data")
        logger.debug(f"Exercise IDs: {exercise_ids}")
        
        return {
            'exercise_ids': exercise_ids,
            'attempts': decrypted_data.get('results', [])
        }
        
    except Exception as e:
        logger.error(f"Error getting attempts for task {task_id}: {str(e)}")
        logger.exception("Full traceback:")
        return None

def load_question_set_context(task_id):
    """Load context for a question set before creating an attempt."""
    url = f"https://www.mathmatize.com/api/mm/question-sets/{task_id}/load_context/"
    headers = get_auth_headers({
        "accept": "application/json",
        "content-type": "application/json",
        "x-mm-version": "7.2.2"
    })
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to load context for task {task_id}: {response.status_code}")
            return False
        
        logger.debug(f"Successfully loaded context for task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Error loading context for task {task_id}: {str(e)}")
        return False

def create_attempt(task_id):
    """Create a new attempt for a task."""
    url = "https://www.mathmatize.com/api/mm/attempts/"
    base_headers = {
        "accept": "application/json; version=1",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "priority": "u=1, i",
        "referer": f"https://www.mathmatize.com/c/1508/?task={task_id}",
        "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Microsoft Edge";v="132"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0",
        "x-mm-version": "7.2.2"
    }
    headers = get_auth_headers(base_headers)
    
    # The payload is encrypted in the same way as responses are decrypted
    data = {
        "task": task_id,
        "type": "AS"  # AS = Assignment
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            logger.error(f"Failed to create attempt for task {task_id}: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
            
        data = response.json()
        decrypted_data = decrypt_response(data)
        
        if not isinstance(decrypted_data, dict):
            logger.error(f"Invalid response data when creating attempt for task {task_id}")
            return None
            
        attempt_id = decrypted_data.get('id')
        if not attempt_id:
            logger.error(f"No attempt ID in response for task {task_id}")
            return None
            
        logger.info(f"Created attempt {attempt_id} for task {task_id}")
        return attempt_id
        
    except Exception as e:
        logger.error(f"Error creating attempt for task {task_id}: {str(e)}")
        logger.exception("Full traceback:")
        return None

def get_exercise_data(exercise_id):
    """Get exercise data with decryption."""
    url = f"https://www.mathmatize.com/api/mm/exercises/{exercise_id}/"
    headers = get_auth_headers({
        "accept": "application/json",
        "content-type": "application/json",
        "x-mm-version": "7.2.2"
    })
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to get exercise {exercise_id}: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
        
        data = response.json()
        logger.debug(f"Raw exercise data for {exercise_id}: {json.dumps(data)[:500]}...")
        
        decrypted_data = decrypt_response(data)
        logger.debug(f"Decrypted exercise data for {exercise_id}: {json.dumps(decrypted_data)[:500]}...")
        
        if not isinstance(decrypted_data, dict):
            logger.error(f"Decrypted data is not a dictionary for exercise {exercise_id}")
            return None

        # Extract relevant fields from decrypted data
        exercise_info = {
            'id': exercise_id,
            'title': decrypted_data.get('title'),
            'description': decrypted_data.get('description'),
            'type': decrypted_data.get('type'),
            'points': decrypted_data.get('points'),
            'content': decrypted_data.get('content'),
            'solution': decrypted_data.get('solution'),
            'metadata': decrypted_data.get('metadata'),
            'raw_data': decrypted_data  # Include full decrypted data
        }
        
        # Only include non-None values
        exercise_info = {k: v for k, v in exercise_info.items() if v is not None}
        
        logger.debug(f"Extracted exercise info: {json.dumps(exercise_info)[:500]}...")
        return exercise_info
        
    except Exception as e:
        logger.error(f"Error getting exercise {exercise_id}: {str(e)}")
        logger.exception("Full traceback:")
        return None

def main():
    # Get credentials from environment variables
    username = os.getenv('MATHMATIZE_USERNAME')
    password = os.getenv('MATHMATIZE_PASSWORD')
    
    if not all([username, password]):
        logger.error("Missing credentials in .env file")
        return
        
    # Set debug logging
    logger.setLevel(logging.DEBUG)
        
    # Authenticate
    if not authenticate(username, password):
        logger.error("Authentication failed")
        return
    
    # Get assignments
    assignments = get_assignments()
    if not assignments:
        logger.error("No assignments found")
        return

    logger.info(f"Successfully retrieved {len(assignments)} assignments")
    
    # Create output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    assignments_file = f"mathmatize_assignments_{timestamp}.txt"
    exercises_file = f"mathmatize_exercises_{timestamp}.json"
    
    # Write assignments summary
    with open(assignments_file, 'w', encoding='utf-8') as f:
        f.write("MATHMATIZE ASSIGNMENTS\n")
        f.write("=" * 50 + "\n\n")
        for assignment in assignments:
            f.write(f"Name: {assignment.get('name', 'N/A')}\n")
            f.write(f"Description: {assignment.get('description', 'N/A')}\n")
            f.write(f"Due Date: {assignment.get('due_date', 'No due date')}\n")
            f.write(f"Status: {assignment.get('status', 'N/A')}\n")
            f.write(f"Points: {assignment.get('points_earned', 0)}/{assignment.get('points', 0)}\n")
            f.write(f"Completion: {assignment.get('completion', 'N/A')}\n")
            f.write("\n")
    
    # Get and save exercise data
    all_exercises = {}
    for assignment in assignments:
        task_id = assignment.get('id')
        if not task_id:
            logger.warning(f"Skipping assignment with no ID: {assignment.get('name', 'N/A')}")
            continue
            
        logger.info(f"Processing assignment: {assignment.get('name', 'N/A')} (ID: {task_id})")
        
        # Get attempts to find exercise IDs
        attempts_data = get_attempts(task_id)
        if not attempts_data:
            logger.warning(f"No attempts data found for task {task_id}")
            continue
            
        # Extract exercise IDs from attempts
        exercise_ids = attempts_data.get('exercise_ids', [])
        logger.info(f"Found {len(exercise_ids)} unique exercise IDs for task {task_id}")
        
        # Get exercise data
        exercises = []
        for ex_id in exercise_ids:
            logger.info(f"Fetching exercise {ex_id}")
            ex_data = get_exercise_data(ex_id)
            if ex_data:
                exercises.append(ex_data)
                logger.debug(f"Added exercise data: {json.dumps(ex_data)[:500]}...")
            else:
                logger.warning(f"Failed to get data for exercise {ex_id}")
        
        if exercises:
            logger.info(f"Adding {len(exercises)} exercises for task {task_id}")
            all_exercises[task_id] = {
                'name': assignment.get('name'),
                'exercises': exercises,
                'attempts': attempts_data.get('attempts', [])
            }
        else:
            logger.warning(f"No exercises found for task {task_id}")
    
    if not all_exercises:
        logger.error("No exercise data collected!")
        return
    
    # Save exercise data with indentation for readability
    try:
        logger.info(f"Writing exercise data to {exercises_file}")
        with open(exercises_file, 'w', encoding='utf-8') as f:
            json.dump(all_exercises, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully wrote exercise data")
        
        # Verify the file was written correctly
        with open(exercises_file, 'r', encoding='utf-8') as f:
            verification_data = json.load(f)
            logger.info(f"Verified JSON file contains {len(verification_data)} assignments")
            
    except Exception as e:
        logger.error(f"Error writing exercise data: {str(e)}")
        logger.exception("Full traceback:")
        return
    
    logger.info(f"Assignments saved to {assignments_file}")
    logger.info(f"Exercise data saved to {exercises_file}")
    logger.info(f"Number of assignments processed: {len(all_exercises)}")
    total_exercises = sum(len(data['exercises']) for data in all_exercises.values())
    logger.info(f"Total number of exercises: {total_exercises}")

if __name__ == "__main__":
    main()