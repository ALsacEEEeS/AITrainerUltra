"""
AITrainerUltra Build Tool - Package into single .exe
Usage:
    python build_exe.py
    python build_exe.py --skip-frontend
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"


def clean():
    for d in [DIST, BUILD / "pyinstaller" / "dist", BUILD / "pyinstaller" / "build"]:
        if d.exists():
            shutil.rmtree(d)
    print("  [OK] cleaned build dirs")


def build_frontend():
    fd = ROOT / "frontend" / "dist"
    if fd.exists() and fd.joinpath("index.html").exists():
        print("  [OK] frontend already built, skip")
        return
    print("  building frontend...")
    result = subprocess.run(
        "npm run build", cwd=str(ROOT / "frontend"),
        shell=True, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  [OK] frontend built")
    else:
        print("  [WARN] frontend build failed, api-only mode:", result.stderr[-200:])


def create_entry():
    p = BUILD / "pyinstaller" / "run_server.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r'''"""AITrainerUltra Standalone"""
import os, sys, time, webbrowser, threading, uvicorn
from pathlib import Path
def open_browser():
    time.sleep(2)
    try: webbrowser.open("http://127.0.0.1:8000")
    except: pass
def main():
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent.parent.parent
    sd = base / "frontend" / "dist"
    if not sd.exists(): sd = base / "dist"
    os.environ["AITRAINER_STATIC_DIR"] = str(sd)
    print("AITrainerUltra v2.0")
    print("Server: http://127.0.0.1:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    from backend.api.server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
if __name__ == "__main__": main()
''', encoding="utf-8")
    print("  [OK] entry created")


def create_spec():
    sp = BUILD / "pyinstaller" / "aitrainer.spec"
    fd = str(ROOT / "frontend" / "dist").replace("\\", "/")
    bd = str(ROOT / "backend").replace("\\", "/")
    sp.write_text(f'''# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Auto-collect all submodules from key packages
all_pkgs = [
    'fastapi', 'starlette', 'uvicorn', 'pydantic', 'websockets',
    'multipart', 'anyio', 'sniffio', 'httptools', 'h11', 'wsproto',
]
extra_hidden = []
for pkg in all_pkgs:
    try:
        extra_hidden.extend(collect_submodules(pkg))
    except Exception:
        pass

a = Analysis(
    ['run_server.py'], pathex=[], binaries=[],
    datas=[(r'{fd}', 'frontend/dist'), (r'{bd}', 'backend')],
    hiddenimports=[
        'backend.api.server','backend.api.routes','backend.api.websocket',
        'backend.api.schemas','backend.core.config','backend.core.engine',
        'backend.core.events','backend.core.registry','backend.core.experiment',
        'backend.core.hpo','backend.core.pipeline','backend.core.plugin',
        'backend.core.scheduler','backend.models.base','backend.utils.device',
        'backend.utils.layers','backend.utils.optimizations',
        'backend.utils.checkpoint','backend.utils.logging','backend.utils.metrics',
        'backend.data.dataset','backend.data.processor','backend.data.generator',
        'backend.data.dataset_manager','backend.data.tokenizer_utils',
    ] + extra_hidden,
    excludes=['tkinter','matplotlib','scipy','cv2','PIL.ImageShow','notebook','jupyter','test','tests'],
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name='AITrainerUltra', debug=False, strip=False, upx=True, console=True,
    disable_windowed_tracker=False, argv_emulation=False)
''', encoding="utf-8")
    print("  [OK] spec created")


def run_pyinstaller(quick=False):
    print("\nRunning PyInstaller (5-10 min)...")
    os.chdir(str(BUILD / "pyinstaller"))
    cmd = "python -m PyInstaller aitrainer.spec --clean --noconfirm"
    if quick:
        cmd += " --strip --noupx"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("  [WARN] stderr:", r.stderr[-400:])
    exe = Path("dist") / "AITrainerUltra.exe"
    if exe.exists():
        print(f"  [OK] built: {exe.stat().st_size / 1024 / 1024:.0f} MB")
    else:
        print("  [FAIL] exe not found")
        sys.exit(1)


def copy_result():
    DIST.mkdir(exist_ok=True)
    src = BUILD / "pyinstaller" / "dist" / "AITrainerUltra.exe"
    if src.exists():
        dst = DIST / "AITrainerUltra.exe"
        shutil.copy2(src, dst)
        print(f"  [OK] output: {dst} ({dst.stat().st_size / 1024 / 1024:.0f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-frontend", action="store_true")
    args = parser.parse_args()

    print("=" * 45)
    print("  AITrainerUltra Build Tool")
    print("=" * 45)

    clean()
    if not args.skip_frontend:
        build_frontend()
    create_entry()
    create_spec()
    run_pyinstaller()
    copy_result()
    print("\nDONE! dist/AITrainerUltra.exe")


if __name__ == "__main__":
    main()
