"""
AITrainerUltra — 一键启动脚本
One command to start both backend and frontend.

新增: --setup 可选虚拟环境初始化
新增: 自动释放端口 (无需手动杀掉旧进程)

Usage:
    python start.py            # Start both (dev mode)
    python start.py --backend  # Backend only
    python start.py --frontend # Frontend only
    python start.py --prod     # Production build + backend
    python start.py --setup    # 首次设置 (创建虚拟环境 + 安装依赖)
    python start.py --skip-venv  # 跳过虚拟环境检测
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parent
VENV_DIR = ROOT / "venv"


def free_port(port: int = 8000) -> None:
    """释放指定端口 — 杀掉占用该端口的进程，防止端口冲突。"""
    if os.name != "nt":
        # Linux/macOS: use fuser
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        return

    # Windows: use netstat + taskkill
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                pid = parts[-1] if parts else ""
                if pid.isdigit():
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True, timeout=5,
                    )
                    print(f"  [Port] 释放端口 {port} (PID {pid})")
    except Exception as e:
        print(f"  [Port] 自动释放端口失败: {e}")

    # 额外检查: 确保端口确实已释放
    time.sleep(0.5)


def print_banner():
    banner = """
    ╔══════════════════════════════════════════╗
    ║        AITrainerUltra 2.0               ║
    ║   Multi-Model AI Training Framework      ║
    ╚══════════════════════════════════════════╝
    """
    print(banner)


def check_venv() -> bool:
    """Check if we're inside a virtual environment."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)


def ensure_venv(skip_venv: bool = False) -> None:
    """Create virtual environment if it doesn't exist and we're not in one."""
    if skip_venv:
        return

    if check_venv():
        print("  [Setup] 已在虚拟环境中运行")
        return

    if VENV_DIR.exists():
        print(f"  [Setup] 检测到虚拟环境: {VENV_DIR}")
        venv_python = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python"
        if venv_python.exists():
            print("  [Setup] 使用虚拟环境的 Python 重新启动...")
            args = [str(venv_python)] + sys.argv + ["--skip-venv"]
            os.execv(str(venv_python), args)
        return

    # Ask to create venv
    print("")
    print("  ────────────────────────────────────────────")
    print("  🔧 是否创建 Python 虚拟环境来隔离依赖？")
    print("     虚拟环境可以避免包冲突，推荐开启。")
    print("     以后可通过 --skip-venv 跳过此步骤。")
    print("  ────────────────────────────────────────────")

    try:
        response = input("  创建虚拟环境? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        response = "y"

    if response in ("", "y", "yes"):
        print(f"  [Setup] 创建虚拟环境: {VENV_DIR}")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [Setup] 虚拟环境创建失败: {result.stderr}")
            print("  [Setup] 使用系统 Python 继续...")
            return

        venv_python = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python"

        print("  [Setup] 安装基础依赖...")
        req_file = ROOT / "backend" / "requirements.txt"
        if req_file.exists():
            result = subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(req_file)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print("  [Setup] 依赖安装完成 ✓")
            else:
                print(f"  [Setup] 部分依赖安装失败: {result.stderr[-200:]}")

        # Re-run with venv python
        print("  [Setup] 使用虚拟环境重新启动...")
        args = [str(venv_python)] + sys.argv + ["--skip-venv"]
        os.execv(str(venv_python), args)
    else:
        print("  [Setup] 跳过虚拟环境，使用系统 Python (可用 --setup 重新开启)")


def start_backend():
    """Start FastAPI backend server."""
    print("  [Backend] Starting uvicorn on http://127.0.0.1:8000")
    print("  [Backend] API docs at http://127.0.0.1:8000/docs")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.server:app",
         "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_frontend_dev():
    """Start Vite dev server for frontend."""
    npm = "npm.cmd" if os.name == "nt" else "npm"
    print("  [Frontend] Starting Vite dev server on http://127.0.0.1:5173")
    return subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(ROOT / "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def build_frontend():
    """Build frontend for production."""
    npm = "npm.cmd" if os.name == "nt" else "npm"
    print("  [Frontend] Building for production...")
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=str(ROOT / "frontend"),
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  [Frontend] Build complete")
    else:
        print(f"  [Frontend] Build failed: {result.stderr[-200:]}")
        sys.exit(1)


def stream_output(proc, prefix):
    """Stream subprocess output line by line."""
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"  {prefix} {text}")


def main():
    # 自动释放端口，防止旧进程占用导致启动失败
    free_port(8000)

    parser = argparse.ArgumentParser(description="AITrainerUltra launcher")
    parser.add_argument("--backend", action="store_true", help="Backend only")
    parser.add_argument("--frontend", action="store_true", help="Frontend only")
    parser.add_argument("--prod", action="store_true", help="Production mode (build frontend first)")
    parser.add_argument("--setup", action="store_true", help="首次设置 (创建虚拟环境 + 安装依赖)")
    parser.add_argument("--skip-venv", action="store_true", help="跳过虚拟环境检测")
    args = parser.parse_args()

    print_banner()

    # Virtual environment setup
    if args.setup:
        ensure_venv(skip_venv=False)
        print("\n  [Setup] 完成！重新运行 python start.py 启动服务")
        return
    else:
        ensure_venv(skip_venv=args.skip_venv)

    mode = "production" if args.prod else "development"
    print(f"  Mode: {mode}\n")

    procs = []

    do_backend = args.backend or not args.frontend
    do_frontend = args.frontend or not args.backend

    if args.prod:
        build_frontend()

    if do_backend:
        p = start_backend()
        procs.append(("Backend", p))

    if do_frontend and not args.prod:
        p = start_frontend_dev()
        procs.append(("Frontend", p))

    if not procs:
        print("  Nothing to start. Use --backend and/or --frontend")
        return

    print("\n  ────────────────────────────────────────────")
    print("  🚀 AITrainerUltra is running!")
    if do_backend:
        print("     Backend:  http://127.0.0.1:8000")
        print("     API Docs: http://127.0.0.1:8000/docs")
    if do_frontend and not args.prod:
        print("     Frontend: http://127.0.0.1:5173")
    if args.prod:
        print("     Frontend: served by FastAPI at /")
    print("  ────────────────────────────────────────────")
    print("  Press Ctrl+C to stop all services\n")

    import threading
    threads = []
    for name, proc in procs:
        t = threading.Thread(target=stream_output, args=(proc, f"[{name}]"), daemon=True)
        t.start()
        threads.append(t)

    try:
        for name, proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        for name, proc in procs:
            proc.terminate()
        print("  All services stopped.")


if __name__ == "__main__":
    main()
