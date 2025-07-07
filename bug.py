import os
from datetime import datetime

CSV_FOLDER = "csv"
os.makedirs(CSV_FOLDER, exist_ok=True)
last_updated_file = os.path.join(CSV_FOLDER, 'last_updated.txt')
with open(last_updated_file, 'w') as f:
    f.write(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
print("Created:", last_updated_file)
