"""
╔══════════════════════════════════════════════════════════════════════╗
║  Antigravity IDE — Auto-Config Generator                            ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Automatski generiše sve konfiguracije za:
  - Gemini CLI
  - Cline (VS Code extension)
  - VS Code settings
  - Environment variables
  - A2A server endpoints
"""

import json, os, sys, shutil
from pathlib import Path

BASE = Path(__file__).parent.parent


def setup_env():
    """Kreira .env iz .env.example ako ne postoji."""
    env_example = BASE / ".env.example"
    env_file = BASE / ".env"
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print(f"[OK] .env kreiran iz .env.example")
        print(f"     Edituj {env_file} i dodaj API ključeve!")
    elif env_file.exists():
        print(f"[OK] .env već postoji: {env_file}")
    return str(env_file)


def setup_vscode():
    """Verifikuje VS Code settings."""
    settings = BASE / ".vscode" / "settings.json"
    if settings.exists():
        print(f"[OK] VS Code settings: {settings}")
    else:
        print(f"[WARN] VS Code settings ne postoji!")
    return str(settings)


def setup_cline():
    """Verifikuje Cline rules."""
    rules = BASE / ".clinerules"
    if rules.exists():
        print(f"[OK] Cline rules: {rules}")
    else:
        print(f"[WARN] .clinerules ne postoji!")
    return str(rules)


def setup_dirs():
    """Kreira sve potrebne direktorijume."""
    dirs = [
        "logs", "chroma_db", "knowledge_base", "kaggle_insights",
        "models", "src/generated", "data/processed", "reports",
        ".agent", "notebooks", "tests",
    ]
    for d in dirs:
        p = BASE / d
        p.mkdir(parents=True, exist_ok=True)
    print(f"[OK] {len(dirs)} direktorijuma kreirano/verifikovano")


def setup_gitignore():
    """Kreira .gitignore."""
    gi = BASE / ".gitignore"
    content = """# Usisivac V6
.env
__pycache__/
*.pyc
chroma_db/
models/*.npz
*.egg-info/
dist/
build/
.agent/
logs/*.jsonl
data/
notebooks/.ipynb_checkpoints/
"""
    gi.write_text(content)
    print(f"[OK] .gitignore: {gi}")


def generate_gemini_config():
    """Generiše Gemini CLI konfiguraciju."""
    config = {
        "project": "Usisivac-V6",
        "protocol": "Trinity Protocol",
        "agents": {
            "orchestrator": "http://localhost:8081",
            "coder": "http://localhost:8082",
            "guardian": "http://localhost:8083",
            "researcher": "http://localhost:8084",
            "critic": "http://localhost:8085",
            "cleaner": "http://localhost:8086",
            "feature": "http://localhost:8087",
            "relay": "http://localhost:8088",
        },
        "commands": {
            "start_all": "python orchestrator/a2a_servers.py",
            "run_once": "python orchestrator/orchestrator.py '{problem}' --once",
            "run_loop": "python orchestrator/orchestrator.py '{problem}' --delay 30",
            "check_log": "tail -f logs/work_log.md",
            "check_drift": "python -c \"from guardian.guardian import verify_proof_registry; print(verify_proof_registry())\"",
            "rag_stats": "python -c \"from core.rag_engine import stats; print(stats())\"",
        },
        "anti_simulation": {
            "enabled": True,
            "proof_registry": "logs/proof_registry.jsonl",
            "work_log": "logs/work_log.md",
            "drift_threshold": 0.4,
        },
    }

    fp = BASE / "config" / "gemini_config.json"
    fp.write_text(json.dumps(config, indent=2))
    print(f"[OK] Gemini config: {fp}")
    return config


def run_full_setup():
    """Pokreće kompletnu konfiguraciju."""
    print("╔══════════════════════════════════════════╗")
    print("║  Antigravity IDE — Setup Starting...     ║")
    print("╚══════════════════════════════════════════╝\n")

    setup_dirs()
    setup_env()
    setup_vscode()
    setup_cline()
    setup_gitignore()
    generate_gemini_config()

    print("\n╔══════════════════════════════════════════╗")
    print("║  Setup Complete!                         ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Next steps:                             ║")
    print("║  1. Edit .env with your API keys         ║")
    print("║  2. pip install -r requirements.txt      ║")
    print("║  3. python config/antigravity_setup.py   ║")
    print("║  4. python orchestrator/a2a_servers.py   ║")
    print("║  5. python orchestrator/orchestrator.py  ║")
    print("╚══════════════════════════════════════════╝")


if __name__ == "__main__":
    run_full_setup()
