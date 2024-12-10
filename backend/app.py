# app.py

import sys
from pathlib import Path
import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for cross-origin resource sharing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from llm.llm_query import recommend_songs

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes and allow requests from http://localhost:3000
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/recommend-by-genre', methods=['POST'])
def recommend_by_genre():
    """
    Handles POST requests for song recommendations.
    Expects JSON with a 'query' field that specifies the user's song preferences.
    """
    data = request.get_json()
    user_query = data.get('query')

    logger.info(f"Received query: {user_query}")

    if not user_query:
        logger.warning("No query provided.")
        return jsonify({"error": "No query provided"}), 400

    try:
        recommendations = recommend_songs(user_query)
        if recommendations.empty:
            logger.info("No recommendations found.")
            return jsonify({"message": "No recommendations found for the given query."}), 200

        # Convert DataFrame to list of dicts for JSON serialization
        results = recommendations.to_dict('records')
        logger.info(f"Returning {len(results)} recommendations.")
        return jsonify({"recommendations": results}), 200
    except Exception as e:
        logger.error(f"Error in recommend_by_genre: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """
    Home route to confirm the API is up and running.
    """
    return jsonify({"message": "Welcome to the Song Recommendation API!"}), 200

if __name__ == '__main__':
    # Ensure the .env is loaded
    env_path = Path(__file__).parent / '.env'    
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.info(".env file loaded.")
    else:
        logger.error(".env file not found. Please ensure it exists in the llm directory.")
        sys.exit(1)

    # Run the Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)