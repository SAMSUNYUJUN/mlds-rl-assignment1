"""Package reproducible deliverables, excluding caches, venv and redundant checkpoints."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','.uv-cache','.python','__pycache__','tmp'}

def main():
    dest=ROOT/'output/assignment1_Yujun_Sun.zip'; dest.parent.mkdir(exist_ok=True)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or any(x in SKIP for x in p.relative_to(ROOT).parts): continue
        if p.suffix in {'.zip','.pyc','.lock','.tmp'} or p.name in {'checkpoint.pt','process.json'}: continue
        if p.suffix=='.pt' and any(part.startswith('pilot_') for part in p.relative_to(ROOT).parts): continue
        files.append(p)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='MANIFEST.json'}
    (ROOT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.name!='MANIFEST.json': z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'MANIFEST.json','MANIFEST.json')
    print(dest, dest.stat().st_size, 'bytes')

if __name__=='__main__': main()
