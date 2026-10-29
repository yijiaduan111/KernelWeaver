#!/usr/bin/env python3
"""Check the public checkout for secrets and accidental generated artifacts.

Only filenames and line numbers are printed for suspected secrets.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [re.compile(rb'sk-[A-Za-z0-9_-]{20,}'),
            re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}'),
            re.compile(rb'github_pat_[A-Za-z0-9_]{25,}'),
            re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]


def main():
    files = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    findings = []
    for name in sorted(set(files.decode().split('\0'))- {''}):
        path = ROOT/name
        if not path.is_file() or path.is_symlink():
            continue
        if (path.name.startswith('.env') and path.name!='.env.example') or name.startswith(('runs/','models/','.runtime/','external/')):
            if name!='runs/.gitkeep':
                findings.append((name,0,'private or generated artifact'))
        data = path.read_bytes()
        for pattern in PATTERNS:
            for match in pattern.finditer(data):
                findings.append((name,data.count(b'\n',0,match.start())+1,'suspected credential'))
    for name,line,reason in findings:
        print(f'{name}:{line}: {reason}')
    if findings:
        return 1
    print('Public checkout checks passed (current files only; not a Git-history audit).')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
