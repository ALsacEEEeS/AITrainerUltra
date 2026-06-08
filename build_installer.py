"""
AITrainerUltra v2.1.0 - Build Installer
One-command build: frontend > PyInstaller .exe > Inno Setup installer

Usage:
    python build_installer.py              # Full build
    python build_installer.py --quick      # Skip UPX (faster)
    python build_installer.py --portable   # Portable zip only (no installer)
    python build_installer.py --no-frontend  # Skip frontend build
"""

import os, sys, shutil, subprocess, argparse, time, zipfile, glob
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DIST = ROOT / "dist"
BUILD = ROOT / "build"
PI_DIR = BUILD / "pyinstaller"


def info(msg):
    print(f"  - {msg}")


def ok(msg):
    print(f"  OK {msg}")


def fail(msg):
    print(f"  FAIL {msg}")


def run(cmd, cwd=ROOT, timeout=600, capture=True):
    print(f"  $ {cmd}")
    kw = dict(cwd=str(cwd), shell=True, capture_output=capture, text=True, timeout=timeout)
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0 and capture:
        for line in result.stderr.strip().splitlines()[-5:]:
            print(f"    {line.strip()}")
    return result


def check_deps():
    """Auto-install missing Python build deps."""
    print("\n  -- Checking build dependencies --")
    missing = []
    for mod in ["PyInstaller", "fastapi", "uvicorn", "pydantic"]:
        try:
            __import__(mod.replace("-", "_"))
        except ImportError:
            missing.append(mod)
    if missing:
        info(f"Installing missing deps: {', '.join(missing)}")
        r = run(f"pip install {' '.join(missing)} --quiet", timeout=120)
        if r.returncode != 0:
            fail(f"Failed to install deps: {r.stderr[-200:]}")
            return False
    ok("All build dependencies installed")
    return True


def build_frontend():
    """Build React frontend to dist/."""
    print("\n  -- Building frontend --")
    fd = ROOT / "frontend" / "dist"
    if fd.exists() and (fd / "index.html").exists():
        ok("Frontend already built, skipping (use --no-frontend to force skip)")
        return True

    info("Installing npm deps...")
    r = run("npm install --silent", ROOT / "frontend", timeout=120)
    if r.returncode != 0:
        fail(f"npm install failed: {r.stderr[-200:]}")
        return False

    info("Building frontend...")
    r = run("npm run build", ROOT / "frontend", timeout=120)
    if r.returncode != 0:
        if fd.exists():
            ok("Frontend build had warnings but output exists")
        else:
            fail("Frontend build failed")
            return False

    size = sum(f.stat().st_size for f in fd.rglob("*") if f.is_file())
    ok(f"Frontend built: {size / 1024 / 1024:.1f} MB")
    return True


def create_entry():
    """Create the PyInstaller entry point."""
    print("\n  -- Creating entry point --")
    PI_DIR.mkdir(parents=True, exist_ok=True)
    entry = PI_DIR / "run_server.py"
    entry.write_text(r'''"""AITrainerUltra v2.1.0 Standalone"""
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
    print("AITrainerUltra v2.1.0")
    print("Server: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop")
    threading.Thread(target=open_browser, daemon=True).start()
    from backend.api.server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
if __name__ == "__main__": main()
''', encoding="utf-8")
    ok("Entry point created")
    return True


def create_spec(quick=False):
    """Create PyInstaller .spec file."""
    print("\n  -- Creating PyInstaller spec --")
    fd = str(ROOT / "frontend" / "dist").replace("\\", "/")
    bd = str(ROOT / "backend").replace("\\", "/")

    spec = PI_DIR / "aitrainer.spec"
    upx_val = "False" if quick else "True"
    spec.write_text(f'''# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

all_pkgs = [
    'fastapi', 'starlette', 'uvicorn', 'pydantic', 'websockets',
    'multipart', 'anyio', 'sniffio', 'httptools', 'h11', 'wsproto',
    'requests', 'httpx', 'yaml', 'jinja2',
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
        'backend.core.scheduler','backend.core.recipes',
        'backend.models.base','backend.utils.device',
        'backend.utils.layers','backend.utils.optimizations',
        'backend.utils.checkpoint','backend.utils.logging','backend.utils.metrics',
        'backend.data.dataset','backend.data.processor','backend.data.generator',
        'backend.data.dataset_manager','backend.data.tokenizer_utils',
        'backend.models.llm_trainer','backend.models.gpt_trainer',
        'backend.models.bert_trainer','backend.models.cnn_trainer',
        'backend.models.rnn_trainer','backend.models.lstm_trainer',
        'backend.models.lora_trainer','backend.models.qlora_trainer',
        'backend.models.moe_trainer','backend.models.multimodal_trainer',
        'backend.models.lcm_trainer','backend.models.scratch_trainer',
        'backend.models.whisper_trainer','backend.models.diffusion_trainer',
        'backend.models.t5_trainer','backend.models.phi_trainer',
        'backend.models.detr_trainer','backend.models.embedding_trainer',
        'backend.models.sam_trainer','backend.models.video_trainer',
        'backend.models.serving','backend.models.model_loader','backend.models.model_manager',
        'backend.models.evaluator',
    ] + extra_hidden,
    excludes=['tkinter','matplotlib','scipy','cv2','PIL.ImageShow',
              'notebook','jupyter','test','tests','ipython',
              'torchvision','torchaudio'],
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name='AITrainerUltra', debug=False, strip=False,
    upx={upx_val}, console=True,
    disable_windowed_tracker=False, argv_emulation=False)
''', encoding="utf-8")
    ok("PyInstaller spec created")
    return True


def run_pyinstaller(quick=False):
    """Execute PyInstaller."""
    print("\n  -- Running PyInstaller (3-10 min) --")
    info("This bundles Python, all deps, and frontend into a single .exe")
    os.chdir(str(PI_DIR))
    cmd = "python -m PyInstaller aitrainer.spec --clean --noconfirm"
    start = time.time()
    r = run(cmd, cwd=PI_DIR, timeout=600)
    elapsed = time.time() - start
    if r.returncode != 0:
        fail(f"PyInstaller failed after {elapsed:.0f}s")
        return False
    exe = Path("dist") / "AITrainerUltra.exe"
    if exe.exists():
        mb = exe.stat().st_size / 1024 / 1024
        ok(f"PyInstaller done ({mb:.0f} MB, {elapsed:.0f}s)")
        return True
    fail("AITrainerUltra.exe not found after build")
    return False


def copy_exe():
    """Copy .exe to dist/ at project root."""
    DIST.mkdir(exist_ok=True)
    src = PI_DIR / "dist" / "AITrainerUltra.exe"
    if src.exists():
        dst = DIST / "AITrainerUltra.exe"
        shutil.copy2(src, dst)
        mb = dst.stat().st_size / 1024 / 1024
        ok(f"Output: {dst} ({mb:.0f} MB)")
        return True
    return False


def build_installer():
    """Run Inno Setup to create the installer .exe."""
    print("\n  -- Building Inno Setup installer --")
    iss = ROOT / "installer" / "setup.iss"
    if not iss.exists():
        fail(f"setup.iss not found at {iss}")
        return False

    # Find ISCC.exe — try common paths, then PATH
    candidates = [
        "C:/Program Files (x86)/Inno Setup 6/ISCC.exe",
        "C:/Program Files/Inno Setup 6/ISCC.exe",
        "C:/Program Files (x86)/Inno Setup 5/ISCC.exe",
        "C:/Program Files/Inno Setup 5/ISCC.exe",
    ]
    iscc = None
    for c in candidates:
        if Path(c).exists():
            iscc = c
            break
    if not iscc:
        # Try PATH
        r = subprocess.run("where ISCC.exe 2>nul", shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            iscc = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None

    if not iscc:
        info("Inno Setup not found - skipping installer")
        info("The .exe is at dist/AITrainerUltra.exe (portable, just run it)")
        info("To create an installer, install Inno Setup 6+ from https://jrsoftware.org")
        return False

    info(f"Using: {iscc}")
    r = run(f'"{iscc}" "{iss}"', timeout=120)
    if r.returncode != 0:
        fail("Inno Setup build failed")
        return False

    # Find the installer
    installer_dir = DIST
    installers = list(installer_dir.glob("AITrainerUltra-Installer-*.exe"))
    if installers:
        exe = installers[0]
        mb = exe.stat().st_size / 1024 / 1024
        ok(f"Installer: {exe} ({mb:.0f} MB)")
        return True
    fail("Installer not found after build")
    return False


def build_portable():
    """Create a portable .zip package."""
    print("\n  -- Creating portable zip --")
    exe_path = DIST / "AITrainerUltra.exe"
    if not exe_path.exists():
        fail("AITrainerUltra.exe not found - build it first")
        return False

    zip_name = DIST / f"AITrainerUltra-v2.1.0-portable.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(exe_path, "AITrainerUltra/AITrainerUltra.exe")
        for f in ["README.md", "CHANGELOG.md", "LICENSE", "SECURITY.md",
                   "start.py", "start_aitrainer.bat", ".env.example"]:
            p = ROOT / f
            if p.exists():
                zf.write(p, f"AITrainerUltra/{f}")
        # Add backend requirements
        req = ROOT / "backend" / "requirements.txt"
        if req.exists():
            zf.write(req, "AITrainerUltra/backend/requirements.txt")

    mb = zip_name.stat().st_size / 1024 / 1024
    ok(f"Portable zip: {zip_name} ({mb:.0f} MB)")
    return True


def clean():
    """Clean build artifacts."""
    print("\n  -- Cleaning previous builds --")
    dirs = [DIST, PI_DIR / "dist", PI_DIR / "build"]
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
            info(f"Removed: {d}")


def main():
    parser = argparse.ArgumentParser(description="AITrainerUltra v2.1.0 Build Tool")
    parser.add_argument("--quick", action="store_true", help="Skip UPX compression (faster)")
    parser.add_argument("--no-frontend", action="store_true", help="Skip frontend build")
    parser.add_argument("--portable", action="store_true", help="Build portable zip only (no installer)")
    parser.add_argument("--skip-clean", action="store_true", help="Don't clean previous builds")
    args = parser.parse_args()

    print()
    print("+--------------------------------------+")
    print("|  AITrainerUltra v2.1.0 Build Tool    |")
    print("|  One-command installer builder       |")
    print("+--------------------------------------+")

    start = time.time()

    # 1. Clean
    if not args.skip_clean:
        clean()
    else:
        info("Skipped clean")

    # 2. Check build deps
    if not check_deps():
        sys.exit(1)

    # 3. Build frontend
    if not args.no_frontend:
        if not build_frontend():
            sys.exit(1)
    else:
        info("Skipped frontend build")

    # 4. Create PyInstaller entry & spec
    if not create_entry():
        sys.exit(1)
    if not create_spec(quick=args.quick):
        sys.exit(1)

    # 5. Run PyInstaller
    if not run_pyinstaller(quick=args.quick):
        sys.exit(1)

    # 6. Copy .exe to dist/
    if not copy_exe():
        sys.exit(1)

    # 7a. Build installer or portable zip
    if args.portable:
        build_portable()
    else:
        build_installer()
        # Also create portable zip as fallback
        build_portable()

    elapsed = time.time() - start
    print()
    print("+--------------------------------------+")
    print(f"|  Done! ({elapsed:.0f}s)                  |")
    if (DIST / "AITrainerUltra.exe").exists():
        print("|  - dist/AITrainerUltra.exe          |")
        print("|    (portable - just run it)         |")
    for f in DIST.glob("AITrainerUltra-*-portable.zip"):
        print(f"|  - {f.name:<38s}|")
    for f in DIST.glob("AITrainerUltra-Installer-*.exe"):
        print(f"|  - {f.name:<38s}|")
    print("+--------------------------------------+")


if __name__ == "__main__":
    main()
