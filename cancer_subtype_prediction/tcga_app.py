"""
Flask running blueprints
"""

import logging
from flask import Flask
import os
from dotenv import load_dotenv
from views.index import bp as index_bp
from views.upload_file import bp10

load_dotenv()
app = Flask(__name__)
name = os.getenv("NAME")


app.secret_key = "secreta"  # Replace 'your_secret_key' with a real secret key

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger(__name__)  # Now it should work as intended

log.info("Adding blue prints")
# Register BPs
app.register_blueprint(index_bp)
app.register_blueprint(bp10)
# app.run(host='0.0.0.0', port=1000)

if __name__ == "__main__":
    # app.secret_key = os.urandom(12)
    # app.run(debug = True)
    app.run(host="0.0.0.0", port="1000")
