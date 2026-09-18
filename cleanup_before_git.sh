#!/usr/bin/env bash
set -Eeuo pipefail

# cleanup_before_git.sh
#
# Purpose:
#   Prepare the WikiLLM project for a public Git commit WITHOUT deleting
#   canonical source code or modifying secrets in-place.
#
# Default behavior:
#   - dry-run style checks + safe cleanup of generated/cache files
#   - creates required .gitkeep files
#   - detects possible secrets and absolute local paths
#   - refuses to remove config/loop.env; it should be gitignored and manually reviewed
#
# Usage:
#   chmod +x cleanup_before_git.sh
#   ./cleanup_before_git.sh
#
# Optional:
#   DELETE_GENERATED=1 ./cleanup_before_git.sh
#     additionally removes generated benchmark outputs and benchmark-run artifacts
#
#   FIX_RUN_SH=1 ./cleanup_before_git.sh
#     does NOT rewrite run.sh automatically; only prints matching absolute paths.
#     This variable is reserved for future automated migration.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DELETE_GENERATED="${DELETE_GENERATED:-0}"

section() {
  printf '\n\033[1;36m== %s ==\033[0m\n' "$1"
}

warn() {
  printf '\033[1;33m[WARN]\033[0m %s\n' "$1"
}

ok() {
  printf '\033[1;32m[OK]\033[0m %s\n' "$1"
}

info() {
  printf '\033[1;34m[INFO]\033[0m %s\n' "$1"
}

remove_if_exists() {
  local path="$1"
  if [[ -e "$path" ]]; then
    rm -rf -- "$path"
    info "removed: $path"
  fi
}

section "Project"
printf 'root: %s\n' "$PROJECT_ROOT"

section "Ensure public-safe directory placeholders"

for d in \
  app/outputs \
  llmwiki/raw/articles \
  llmwiki/raw/papers \
  llmwiki/raw/repos \
  llmwiki/raw/data \
  llmwiki/raw/images \
  llmwiki/raw/assets
do
  mkdir -p "$d"
  touch "$d/.gitkeep"
done

ok ".gitkeep files ensured"

section "Remove Python/cache artifacts"

find . -type d \
  \( -name '__pycache__' \
     -o -name '.pytest_cache' \
     -o -name '.mypy_cache' \
     -o -name '.ruff_cache' \
     -o -name '*.egg-info' \) \
  -prune -print -exec rm -rf {} + 2>/dev/null || true

find . -type f \
  \( -name '*.pyc' \
     -o -name '*.pyo' \
     -o -name '*.swp' \
     -o -name '*.swo' \
     -o -name '*.tmp' \
     -o -name '*.temp' \
     -o -name '*.bak' \) \
  -print -delete 2>/dev/null || true

ok "cache/build leftovers removed"

section "Remove local Obsidian workspace state"

remove_if_exists "llmwiki/.obsidian/workspace.json"
remove_if_exists "llmwiki/.obsidian/workspace-mobile.json"
remove_if_exists "llmwiki/.obsidian/cache"
remove_if_exists "llmwiki/.trash"

section "Generated benchmark data"

if [[ "$DELETE_GENERATED" == "1" ]]; then
  remove_if_exists "app/outputs/benchmark"
  remove_if_exists "llmwiki/benchmark-runs"
  remove_if_exists "llmwiki/benchmarks"

  mkdir -p app/outputs
  touch app/outputs/.gitkeep

  ok "generated benchmark artifacts removed"
else
  warn "generated data NOT deleted"
  printf '%s\n' \
    "Set DELETE_GENERATED=1 to remove:" \
    "  app/outputs/benchmark/" \
    "  llmwiki/benchmark-runs/" \
    "  llmwiki/benchmarks/"
fi

section "Runtime secret files"

for f in \
  ".env" \
  "config/loop.env" \
  "config/runtime.env"
do
  if [[ -f "$f" ]]; then
    warn "private runtime file exists: $f"
  fi
done

cat <<'EOF'

These files should remain local and must be excluded by .gitignore.
Do NOT replace real values with fake-looking keys and accidentally commit them.
Prefer an example file such as:

  config/runtime.env.example

with entries like:

  OLLAMA_HOST=http://localhost:11434
  OLLAMA_API_KEY=
  OPENAI_API_KEY=

EOF

section "Secret scan"

SECRET_PATTERN='(OPENAI_API_KEY|OLLAMA_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET_ACCESS_KEY|sk-[A-Za-z0-9_-]{12,}|oa-[A-Za-z0-9_-]{8,}|Bearer[[:space:]]+[A-Za-z0-9._-]{12,})'

if command -v rg >/dev/null 2>&1; then
  if rg -n --hidden \
      --glob '!.git/**' \
      --glob '!app/outputs/**' \
      --glob '!llmwiki/benchmark-runs/**' \
      --glob '!*.zip' \
      "$SECRET_PATTERN" .; then
    warn "possible secrets found above — review before git add"
  else
    ok "no obvious secrets found by regex scan"
  fi
else
  warn "ripgrep (rg) not installed; using grep fallback"
  if grep -RInE \
      --exclude-dir=.git \
      --exclude-dir=__pycache__ \
      --exclude='*.zip' \
      "$SECRET_PATTERN" . 2>/dev/null; then
    warn "possible secrets found above — review before git add"
  else
    ok "no obvious secrets found by regex scan"
  fi
fi

section "Absolute/local path scan"

PATH_PATTERN='(/home/[^/[:space:]]+|/mnt/c/Users/[^/[:space:]]+|[A-Za-z]:\\Users\\[^\\[:space:]]+)'

if command -v rg >/dev/null 2>&1; then
  if rg -n --hidden \
      --glob '!.git/**' \
      --glob '!app/outputs/**' \
      --glob '!llmwiki/benchmark-runs/**' \
      "$PATH_PATTERN" .; then
    warn "machine-specific paths found above"
    printf '%s\n' \
      "Replace hard-coded paths with relative paths or variables such as:" \
      '  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' \
      '  LLMWIKI_ROOT="${LLMWIKI_ROOT:-$PROJECT_ROOT/llmwiki}"'
  else
    ok "no obvious machine-specific absolute paths found"
  fi
fi

section "Large tracked candidates (>10 MiB)"

find . \
  -type f \
  -not -path './.git/*' \
  -size +10M \
  -printf '%s %p\n' 2>/dev/null \
  | sort -nr \
  | awk '{
      mb=$1/1024/1024;
      $1="";
      printf "%.1f MiB%s\n", mb, $0
    }' || true

section "Git status / ignored files"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short || true

  printf '\nIgnored candidates:\n'
  git status --short --ignored 2>/dev/null \
    | grep '^!!' \
    | head -n 100 || true
else
  warn "not currently inside a Git repository"
fi

section "Final pre-commit commands"

cat <<'EOF'
Review before committing:

  git status --short
  git diff --cached

Check ignored secret/config files:

  git check-ignore -v config/loop.env .env 2>/dev/null || true

Inspect exactly what would be committed:

  git ls-files

If a secret was previously committed, adding .gitignore is NOT enough.
Rotate the credential and remove it from Git history before publishing.

Suggested first staging pass:

  git add \
    .gitignore \
    cleanup_before_git.sh \
    app \
    config \
    docs \
    llmwiki \
    tests \
    README.md \
    pyproject.toml \
    requirements.txt

Then review again:

  git status --short
  git diff --cached
EOF

ok "cleanup/check completed"
