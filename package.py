"""
AITrainerUltra 一键打包脚本
One-command build: frontend → backend → .exe

Usage:
    python package.py              # Full build (frontend + backend + .exe)
    python package.py --quick      # Skip UPX compression (faster)
    python package.py --skip-frontend  # Use existing frontend build
    python package.py --no-exe     # Build frontend + backend only (no .exe)
    python package.py --frontend-only  # Build frontend only
"""

import os
import sys
import shutil
import subprocess
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent
DIST = ROOT / "dist"
BUILD_DIR = ROOT / "build"


def print_header(text: str):
    print()
    print("=" * 50)
    print(f"  {text}")
    print("=" * 50)


def run(cmd: str, cwd: Path = ROOT, timeout: int = 300) -> bool:
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, cwd=str(cwd), shell=True,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"  ❌ FAILED (code {result.returncode})")
        if result.stderr:
            print(f"  Error: {result.stderr[-300:]}")
        return False
    if result.stdout:
        # Print relevant output (last few lines)
        lines = result.stdout.strip().split('\n')
        for line in lines[-3:]:
            if line.strip():
                print(f"  {line.strip()}")
    return True


def build_frontend() -> bool:
    print_header("1/3: 构建前端 (Building Frontend)")
    fd = ROOT / "frontend" / "dist"
    if fd.exists() and fd.joinpath("index.html").exists():
        print("  ⏭️  Frontend already built, skipping (use --skip-frontend to force)")
        return True
    if not run("npm install", ROOT / "frontend", timeout=120):
        return False
    if not run("npm run build", ROOT / "frontend", timeout=120):
        print("  ⚠️  Frontend build failed, will try API-only mode")
        return False
    size = sum(f.stat().st_size for f in fd.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  ✅ Frontend built: {size:.1f} MB")
    return True


def check_deps() -> bool:
    print_header("检查依赖 (Checking Dependencies)")
    # Check Python deps
    missing = []
    for mod in ["fastapi", "uvicorn", "pydantic", "PyInstaller"]:
        try:
            __import__(mod.replace("-", "_"))
        except ImportError:
            missing.append(mod)

    if missing:
        print(f"  ⚠️  缺少模块: {', '.join(missing)}")
        ans = input("  是否自动安装? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return run(f"pip install {' '.join(missing)}", timeout=120)
        else:
            print("  ❌ 请手动安装后重试")
            return False
    print("  ✅ 所有依赖已安装")
    return True


def build_backend_entry() -> bool:
    print_header("2/3: 创建入口文件 (Creating Entry Point)")
    p = BUILD_DIR / "pyinstaller" / "run_server.py"
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
    print("Press Ctrl+C to stop")
    threading.Thread(target=open_browser, daemon=True).start()
    from backend.api.server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
if __name__ == "__main__": main()
''', encoding="utf-8")
    print("  ✅ 入口文件已创建")
    return True


def build_spec(quick: bool = False) -> bool:
    print_header("2/3: 创建 PyInstaller Spec")
    fd = str(ROOT / "frontend" / "dist").replace("\\", "/")
    bd = str(ROOT / "backend").replace("\\", "/")
    sp = BUILD_DIR / "pyinstaller" / "aitrainer.spec"
    sp.write_text(f'''# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

all_pkgs = [
    'fastapi', 'starlette', 'uvicorn', 'pydantic', 'websockets',
    'multipart', 'anyio', 'sniffio', 'httptools', 'h11', 'wsproto',
    'requests', 'httpx', 'yaml',
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
        'backend.models.sam_trainer',
    ] + extra_hidden,
    excludes=['tkinter','matplotlib','scipy','cv2','PIL.ImageShow',
              'notebook','jupyter','test','tests','ipython'],
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
    name='AITrainerUltra', debug=False, strip=False,
    upx={str(not quick).lower()}, console=True,
    disable_windowed_tracker=False, argv_emulation=False)
''', encoding="utf-8")
    print("  ✅ Spec 已创建")
    return True


def run_pyinstaller(quick: bool = False) -> bool:
    print_header("3/3: 打包 (Running PyInstaller)")
    print("  ⏳ 这需要 3-10 分钟...")
    os.chdir(str(BUILD_DIR / "pyinstaller"))
    cmd = "python -m PyInstaller aitrainer.spec --clean --noconfirm"
    if quick:
        cmd += " --strip --noupx"
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"  ❌ PyInstaller failed after {elapsed:.0f}s")
        for line in result.stderr.split('\n')[-10:]:
            if line.strip():
                print(f"     {line.strip()}")
        return False
    exe = Path("dist") / "AITrainerUltra.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / 1024 / 1024
        print(f"  ✅ 打包完成! ({size_mb:.0f} MB, {elapsed:.0f}s)")
        return True
    print("  ❌ exe not found")
    return False


def copy_result() -> bool:
    DIST.mkdir(exist_ok=True)
    src = BUILD_DIR / "pyinstaller" / "dist" / "AITrainerUltra.exe"
    if src.exists():
        dst = DIST / "AITrainerUltra.exe"
        shutil.copy2(src, dst)
        mb = dst.stat().st_size / 1024 / 1024
        print(f"  ✅ 输出: {dst} ({mb:.0f} MB)")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="AITrainerUltra 一键打包工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python package.py                    # 完整打包
  python package.py --quick            # 快速打包 (跳过UPX)
  python package.py --no-exe           # 只构建前端+后端, 不打.exe
  python package.py --frontend-only    # 只构建前端
        """,
    )
    parser.add_argument("--quick", action="store_true", help="快速模式 (跳过UPX压缩)")
    parser.add_argument("--skip-frontend", action="store_true", help="跳过前端构建")
    parser.add_argument("--no-exe", action="store_true", help="不打包exe")
    parser.add_argument("--frontend-only", action="store_true", help="只构建前端")
    args = parser.parse_args()

    print()
    print("╔════════════════════════════════════════╗")
    print("║   AITrainerUltra 一键打包工具          ║")
    print("║   One-Click Package Tool               ║")
    print("╚════════════════════════════════════════╝")

    start = time.time()

    # Step 1: Build frontend
    if args.frontend_only:
        build_frontend()
        print(f"\n  ✨ 完成! 耗时: {time.time() - start:.0f}s")
        return

    if not check_deps():
        sys.exit(1)

    if not build_frontend():
        if not args.skip_frontend:
            print("  ⚠️  前端构建失败, 使用API-only模式继续")
    else:
        print()

    if args.no_exe:
        print(f"\n  ✨ 前端构建完成! 耗时: {time.time() - start:.0f}s")
        print("  运行 python start.py 启动开发服务器")
        return

    # Step 2: Create PyInstaller entry and spec
    if not build_backend_entry():
        sys.exit(1)
    if not build_spec(quick=args.quick):
        sys.exit(1)

    # Step 3: Run PyInstaller
    if not run_pyinstaller(quick=args.quick):
        sys.exit(1)
    if not copy_result():
        sys.exit(1)

    elapsed = time.time() - start
    print()
    print("=" * 50)
    print(f"  ✅ 全部完成! 总耗时: {elapsed:.0f}s")
    print(f"  📦 输出: {DIST / 'AITrainerUltra.exe'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
