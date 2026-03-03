#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

if [[ -f "$HOOK_PATH" ]] && ! grep -q "scripts/run_controller.py" "$HOOK_PATH"; then
  cp "$HOOK_PATH" "$HOOK_PATH.backup"
fi

cat > "$HOOK_PATH" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -d "venv" ]]; then
  source venv/bin/activate
fi

python scripts/run_controller.py --staged --mode pre-commit
EOF

chmod +x "$HOOK_PATH"
echo "Installed pre-commit task controller hook at $HOOK_PATH"
