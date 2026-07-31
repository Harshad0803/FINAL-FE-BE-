cd "c:\Users\adeha\Downloads\Final UI 2_backup"; & "c:/Users/adeha/Downloads/Final UI 2_backup/.venv-1/Scripts/python.exe" "c:/Users/adeha/Downloads/Final UI 2_backup/debug_ollama_compare.py"
$tmp = Join-Path $env:TEMP 'debug_ollama_compare.py'; $script = @"
import inspect
import json
import sys
import urllib.request
from pathlib import Path
