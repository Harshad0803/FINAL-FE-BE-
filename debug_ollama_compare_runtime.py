import inspect
import json
import sys
import urllib.request
from pathlib import Path

root = Path(r'c:/Users/adeha/Downloads/Final UI 2_backup/Credit-Risk-Poc-main')
sys.path.insert(0, str(root))

import llm_providers
import agent2 as agent2_module

print('MODULE_FILE', llm_providers.__file__)
print('COMPLETE_METHOD_FILE', inspect.getsourcefile(llm_providers.OllamaProvider.complete))

provider = llm_providers.OllamaProvider()
print('PROVIDER_MODEL', provider.model)
print('PROVIDER_BASE_URL', provider.base_url)

rules_path = root / 'rag_store' / 'val_mdd_rules.json'
doc_path = root / 'demo_data' / 'flawed_mdd.txt'

rules = json.loads(rules_path.read_text(encoding='utf-8'))
doc_text = doc_path.read_text(encoding='utf-8')
conceptual_rules = [r for r in rules if r.get('stage') == 'conceptual']
agent = agent2_module.Agent2(str(rules_path))
prompt = agent._build_llm_prompt(conceptual_rules, doc_text)

standalone_payload = {
    'model': 'llama3.2:latest',
    'prompt': prompt,
    'stream': False,
    'format': {
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
    },
    'options': {'temperature': 0},
}

real_urlopen = urllib.request.urlopen


def debug_urlopen(req, timeout=None, context=None, *args, **kwargs):
    body_bytes = req.data or b''
    body_text = body_bytes.decode('utf-8') if body_bytes else ''
    print('REQUEST_BODY_START')
    print(body_text)
    print('REQUEST_BODY_END')

    parsed = json.loads(body_text) if body_text else {}
    print('REQUEST_MODEL', parsed.get('model'))
    print('REQUEST_STREAM', parsed.get('stream'))
    print('REQUEST_OPTIONS', parsed.get('options'))
    print('REQUEST_FORMAT', json.dumps(parsed.get('format'), sort_keys=True))
    print('STANDALONE_MODEL', standalone_payload['model'])
    print('STANDALONE_STREAM', standalone_payload['stream'])
    print('STANDALONE_OPTIONS', standalone_payload['options'])
    print('STANDALONE_FORMAT', json.dumps(standalone_payload['format'], sort_keys=True))

    resp = real_urlopen(req, timeout=timeout, context=context, *args, **kwargs)

    class WrappedResponse:
        def __init__(self, inner):
            self._inner = inner
            self.status = getattr(inner, 'status', 200)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            self._inner.close()
            return False
        def read(self):
            data = self._inner.read().decode('utf-8')
            print('RESPONSE_BODY_START')
            print(data)
            print('RESPONSE_BODY_END')
            return data.encode('utf-8')
        def close(self):
            self._inner.close()

    return WrappedResponse(resp)


urllib.request.urlopen = debug_urlopen
try:
    result = provider.complete(prompt)
    print('PROVIDER_RESULT', result)
finally:
    urllib.request.urlopen = real_urlopen
