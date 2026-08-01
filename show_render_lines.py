from pathlib import Path
path = Path('aegis-frontend/src/routes/validation.data-quality.tsx')
for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
    if 420 <= i <= 470:
        print(f'{i}: {line}')