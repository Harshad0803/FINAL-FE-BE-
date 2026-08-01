import json
import requests
from pathlib import Path

base = Path('c:/Users/adeha/Downloads/Final UI 2_backup/24-06')
files = {
    'file': ('sample.csv', (base / 'sample.csv').open('rb'), 'text/csv'),
    'mdd_file': ('clean_mdd.txt', (base / 'demo_data' / 'clean_mdd.txt').open('rb'), 'text/plain'),
}
data = {'intake_json': json.dumps({'frameworks': ['RBI', 'IFRS9', 'SS1/23']})}

r = requests.post('http://127.0.0.1:8000/validation/stage2/run', files=files, data=data, timeout=600)
print('STATUS', r.status_code)
print(r.text[:6000])
