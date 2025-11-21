import os
import sys

port = os.environ.get('PORT', '10000')
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Starting gunicorn on port {port}", file=sys.stderr)
print(f"PORT environment variable: {os.environ.get('PORT')}", file=sys.stderr)

bind = f"0.0.0.0:{port}"
workers = 1
timeout = 120
loglevel = "debug"
accesslog = "-"
errorlog = "-"
capture_output = True
