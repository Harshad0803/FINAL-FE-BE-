from pathlib import Path
for rel in ['aegis-frontend/src/routes/validation.data-quality.tsx','aegis-frontend/src/components/check-summary.tsx']:
    print('---', rel, '---')
    path = Path(rel)
    lines = path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines, 1):
        if rel.endswith('validation.data-quality.tsx') and ((220 <= i <= 270) or (900 <= i <= 1025)):
            print(f'{i}: {line}')
        if rel.endswith('check-summary.tsx') and (1 <= i <= 120):
            print(f'{i}: {line}')
