# MathMatize Scraper

This project is a Python-based scraper for interacting with the MathMatize API. It retrieves assignments and exercise data, decrypts API responses, and saves the data to local files.

## Features

- **Authentication:** Uses Firebase authentication to log in with provided credentials.
- **Assignment Retrieval:** Fetches assignments from the MathMatize student API.
- **Data Decryption:** Decrypts response payloads using XOR, Base64 decoding, and JSON parsing.
- **Exercise Data Extraction:** Retrieves and processes detailed exercise data.

## Setup

1. **Clone the Repository**

   Ensure that you have all project files in your workspace:
   - [mathmatize_scraper.py](mathmatize_scraper.py)
   - [.env](.env)
   - [.gitignore](.gitignore)

2. **Environment Variables**

   Create a `.env` file (already provided) with the following credentials:

   ```properties
   # Mathmatize Credentials
   MATHMATIZE_USERNAME=your_username@example.com
   MATHMATIZE_PASSWORD=your_password