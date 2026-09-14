#!/usr/bin/env bash
# Fix Cursor integrated terminal on CentOS 7 (GLIBC 2.17).
# Re-run after every Cursor auto-update that installs a new cursor-server.
#
# Usage:
#   fix-cursor-terminal.sh              # patch + restart cursor-server
#   fix-cursor-terminal.sh --patch-only # patch only (safe while connected)

set -euo pipefail

PATCH_ONLY=0
if [[ "${1:-}" == "--patch-only" ]]; then
  PATCH_ONLY=1
fi

COMPAT_COMMIT="4aa8ff1b7877ed7bd01bcba308698f71a6735380"
BASE="/work/home/nly_2026/.cursor-server/bin/linux-x64"
COMPAT_DIR="${BASE}/${COMPAT_COMMIT}"
COMPAT_PTY="${COMPAT_DIR}/node_modules/node-pty/build/Release/pty.node"
COMPAT_NODE="${COMPAT_DIR}/node"

if [[ ! -f "$COMPAT_PTY" || ! -f "$COMPAT_NODE" ]]; then
  echo "ERROR: compatible reference server missing at $COMPAT_DIR"
  exit 1
fi

if [[ "$PATCH_ONLY" -eq 0 ]]; then
  echo "[1/3] Stopping cursor-server..."
  pkill -u "$(whoami)" -f cursor-server 2>/dev/null || true
  sleep 2
else
  echo "[1/3] Patch-only mode (cursor-server keeps running)"
fi

echo "[2/3] Patching all cursor-server versions..."
patched=0
for dir in "$BASE"/*/; do
  commit="$(basename "$dir")"
  pty="${dir}node_modules/node-pty/build/Release/pty.node"
  [[ -f "$pty" ]] || continue

  if ldd "$pty" 2>&1 | rg -q "GLIBC_2.28.*not found"; then
    echo "  patch pty.node: $commit"
    cp -a "$pty" "${pty}.orig" 2>/dev/null || true
    cp -a "$COMPAT_PTY" "$pty"
    patched=$((patched + 1))
  fi

  cd "$dir"
  if ! ./node -e "try{require('node-pty');process.exit(0)}catch(e){process.exit(1)}" 2>/dev/null; then
    echo "  restore bundled node: $commit"
    rm -f node
    cp -a "$COMPAT_NODE" ./node
    chmod +x ./node
    patched=$((patched + 1))
  fi
done

if [[ "$patched" -eq 0 ]]; then
  echo "  all versions already patched"
fi

echo "[3/3] Verifying servers..."
for dir in "$BASE"/*/; do
  commit="$(basename "$dir")"
  pty="${dir}node_modules/node-pty/build/Release/pty.node"
  [[ -f "$pty" ]] || continue
  cd "$dir"
  ./node -e "try{require('node-pty');console.log('  OK:', '${commit}')}catch(e){console.error('  FAIL:', '${commit}', e.message);process.exit(1)}" 2>&1
done

echo ""
if [[ "$PATCH_ONLY" -eq 1 ]]; then
  echo "Done. Run: Developer: Reload Window, then open terminal (Ctrl+\`)"
else
  echo "Done. Reconnect Cursor and open terminal (Ctrl+\`)"
fi
