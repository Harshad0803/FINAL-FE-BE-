import json
import urllib.request
from pathlib import Path

root = Path(r'c:/Users/adeha/Downloads/Final UI 2_backup/Credit-Risk-Poc-main')
prompt_path = root / 'prompts' / 'conceptual_validation_prompt.txt'
rules_path = root / 'rag_store' / 'val_mdd_rules.json'
doc_path = root / 'demo_data' / 'flawed_mdd.txt'

prompt_template = prompt_path.read_text(encoding='utf-8')
rules = json.loads(rules_path.read_text(encoding='utf-8'))
doc_text = doc_path.read_text(encoding='utf-8')

conceptual_rules = [r for r in rules if r.get('stage') == 'conceptual']
rules_payload = [{
    'index': i,
    'rule_id': str(r.get('id', r.get('check', '?'))),
    'statement': r.get('rule', r.get('flag', '')),
    'principle': r.get('principle', ''),
    'mdd_section_hint': r.get('mdd_section_hint', ''),
} for i, r in enumerate(conceptual_rules)]
prompt = (
    prompt_template
    .replace('{{RULE_COUNT}}', str(len(conceptual_rules)))
    .replace('{{RULE_INDEXES}}', ', '.join(str(i) for i in range(len(conceptual_rules))))
    .replace('{{RULES_JSON}}', json.dumps(rules_payload, indent=2))
    .replace('{{DOCUMENT_TEXT}}', doc_text)
)

schema = {
    'type': 'array',
    'items': {
        'type': 'object',
        'required': ['index', 'rule_id', 'status', 'evidence', 'reasoning'],
        'properties': {
            'index': {'type': 'integer'},
            'rule_id': {'type': 'string'},
            'status': {'type': 'string', 'enum': ['PASS', 'WARN', 'FAIL']},
            'evidence': {'type': 'string'},
            'reasoning': {'type': 'string'},
        },
        'additionalProperties': False,
    },
}

payload = {
    'model': 'llama3.2:latest',
    'prompt': prompt,
    'stream': False,
    'format': schema,
    'options': {'temperature': 0},
}

req = urllib.request.Request('http://127.0.0.1:11434/api/generate', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    data = json.loads(r.read().decode())
    print('RESPONSE_TEXT_START')
    print(data.get('response', ''))
    print('RESPONSE_TEXT_END')
