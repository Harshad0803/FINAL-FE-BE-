import json
import urllib.request

payload = {
    'model': 'llama3.1:latest',
    'prompt': 'Return a JSON array with one verdict object.',
    'stream': False,
    'format': {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'index': {'type': 'integer'},
                'rule_id': {'type': 'string'},
                'status': {'type': 'string', 'enum': ['PASS', 'WARN', 'FAIL']},
                'evidence': {'type': 'string'},
                'reasoning': {'type': 'string'},
            },
            'required': ['index', 'rule_id', 'status', 'evidence', 'reasoning'],
            'additionalProperties': False,
        },
    },
    'options': {'temperature': 0},
}

req = urllib.request.Request(
    'http://localhost:11434/api/generate',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(req, timeout=300) as resp:
    body = resp.read().decode('utf-8')
    print(body)
