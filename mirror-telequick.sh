#!/usr/bin/env bash
# Generate the TeleQuick-branded mirror of this starter.
#
# ClutchCall is CANONICAL here: make every change in this tree and re-run this.
# The mirror is GENERATED and gitignored, never hand-edited, so there is no
# second copy to drift. Same mechanical sed pass the other dual-brand repos use
# (clutchcall-livekit-transport/mirror-telequick.sh), in the same direction:
#
#   ClutchCall -> TeleQuick,  CLUTCHCALL -> TELEQUICK,  clutchcall -> telequick
#
# which rewrites brand identity in one pass (pip package name, python module,
# engine host, env var prefixes, file names) — then brand-named PATHS are
# renamed.
#
#   ./mirror-telequick.sh             # -> .mirror/telequick/
#   OUT=/tmp/x ./mirror-telequick.sh  # somewhere else
#
# ── Why a protected list ─────────────────────────────────────────────────────
# A blanket pass hits strings that only LOOK like the brand. Each of these names
# one real thing that has exactly one name, whichever brand is asking:
#
#   artifacts.clutchcall.dev  the ONE package origin. It serves both brands.
#                             artifacts.telequick.dev is an unclaimed Vercel
#                             subdomain and answers DEPLOYMENT_NOT_FOUND.
#   @clutchcall/admin-sdk     a sibling package. @telequick/admin-sdk is not
#                             published, so the pointer would go nowhere.
#
# Everything else is a real rename: `livekit-plugins-clutchcall` has a
# `livekit-plugins-telequick` twin on the same index, `engine.clutchcall.dev`
# has `engine.telequick.dev`, and the plugin reads BOTH env prefixes
# (CLUTCHCALL_* and TELEQUICK_*), so a renamed variable still works.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/.mirror/telequick}"

PROTECTED=(
  "artifacts.clutchcall.dev"
  "@clutchcall/admin-sdk"
)
# This script itself does not ship in the mirror — it is ClutchCall tooling,
# and a copy inside the generated tree would invite someone to run it there.
EXCLUDE_RE='^mirror-telequick\.sh$'

rm -rf "$OUT"; mkdir -p "$OUT"
# Only TRACKED files, so build output (.venv, __pycache__, .mirror) can never
# leak into a published mirror.
( cd "$HERE" && git ls-files | grep -vE "$EXCLUDE_RE" | tar cf - -T - ) \
  | ( cd "$OUT" && tar xf - )

# Protected tokens are LITERALS, but sed reads them as a regex. Escape every
# BRE metacharacter before handing one to sed.
bre() { printf '%s' "$1" | sed -e 's/[][\.*^$|]/\\&/g'; }

# 1. Protect. \x01<i>\x01 cannot occur in source; the pass cannot see through it.
i=0
for tok in "${PROTECTED[@]}"; do
  find "$OUT" -type f -print0 | xargs -0 sed -i "s|$(bre "$tok")|$(printf '\001%d\001' "$i")|g"
  i=$((i + 1))
done

# 2. Content. The tree is text-only, so a blanket pass is safe.
find "$OUT" -type f -print0 | xargs -0 sed -i \
  -e 's/ClutchCall/TeleQuick/g' \
  -e 's/CLUTCHCALL/TELEQUICK/g' \
  -e 's/clutchcall/telequick/g'

# 3. Restore.
i=0
for tok in "${PROTECTED[@]}"; do
  find "$OUT" -type f -print0 | xargs -0 sed -i "s|$(printf '\001%d\001' "$i")|$tok|g"
  i=$((i + 1))
done

# 4. Paths. Deepest-first so a parent rename cannot invalidate a child's path.
while IFS= read -r p; do
  [ -e "$p" ] || continue
  new="$(dirname "$p")/$(basename "$p" | sed -e 's/ClutchCall/TeleQuick/g' -e 's/clutchcall/telequick/g')"
  [ "$p" = "$new" ] || mv "$p" "$new"
done < <(find "$OUT" -depth -iname '*clutchcall*')

# 5. Nothing brand-crossed may remain except the protected strings.
strip_protected() { # read stdin, blank out every protected token
  local sedargs=()
  for tok in "${PROTECTED[@]}"; do sedargs+=(-e "s|$(bre "$tok")||g"); done
  sed "${sedargs[@]}"
}
# Grep from INSIDE $OUT: an absolute path would itself carry "clutchcall" (this
# checkout's own directory name) and every line would look like a residue.
residual=$(cd "$OUT" && grep -rniI "clutchcall" . 2>/dev/null | strip_protected | grep -i "clutchcall" || true)
if [ -n "$residual" ] || find "$OUT" -iname '*clutchcall*' | grep -q .; then
  echo "err  residual clutchcall tokens/paths in the mirror:" >&2
  [ -n "$residual" ] && printf '%s\n' "$residual" | head -5 >&2
  find "$OUT" -iname '*clutchcall*' | head -5 >&2
  exit 1
fi

echo "mirrored -> $OUT"
echo "  plugin  $(grep -oP 'livekit-plugins-telequick[^"]*' "$OUT/pyproject.toml" | head -1)"
echo "  worker  $(ls "$OUT"/src/*_worker.py)"
