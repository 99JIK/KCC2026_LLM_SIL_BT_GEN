#!/usr/bin/env bash
# Build the BT.CPP v4 stage-4 runtime loader.
#
# Requires:
#   - g++ (C++17)
#   - BehaviorTree.CPP v4 development package installed system-wide
#       Debian/Ubuntu: install from BehaviorTree.CPP repo or vcpkg
#       Headers expected at /usr/local/include/behaviortree_cpp/
#       Library expected at /usr/local/lib/libbehaviortree_cpp.so
#
# This script is idempotent; rerun it after modifying btcpp_loader.cpp.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/src/bt_validator/btcpp_loader.cpp"
OUT="$REPO_ROOT/src/bt_validator/btcpp_loader"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: source not found at $SRC" >&2
  exit 1
fi

# Locate BT.CPP headers and library.
INCLUDE_DIR=""
for cand in /usr/local/include/behaviortree_cpp /usr/include/behaviortree_cpp; do
  if [[ -d "$cand" ]]; then
    INCLUDE_DIR="$(dirname "$cand")"
    break
  fi
done

if [[ -z "$INCLUDE_DIR" ]]; then
  echo "ERROR: BehaviorTree.CPP headers not found." >&2
  echo "Install BehaviorTree.CPP v4 system-wide and re-run." >&2
  echo "  https://github.com/BehaviorTree/BehaviorTree.CPP" >&2
  exit 1
fi

LIB_DIR=""
for cand in /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu; do
  if [[ -f "$cand/libbehaviortree_cpp.so" ]]; then
    LIB_DIR="$cand"
    break
  fi
done

if [[ -z "$LIB_DIR" ]]; then
  echo "ERROR: libbehaviortree_cpp.so not found." >&2
  exit 1
fi

echo "Using headers at $INCLUDE_DIR/behaviortree_cpp"
echo "Using library at $LIB_DIR/libbehaviortree_cpp.so"

g++ \
  -std=c++17 \
  -O2 \
  -Wall -Wextra \
  -I"$INCLUDE_DIR" \
  -L"$LIB_DIR" \
  -o "$OUT" \
  "$SRC" \
  -lbehaviortree_cpp

echo "Built: $OUT"

# Quick smoke test: build a trivial valid BT XML and run the loader on it.
SMOKE_XML="$(mktemp --suffix=.xml)"
trap 'rm -f "$SMOKE_XML"' EXIT
cat > "$SMOKE_XML" <<'EOF'
<root BTCPP_format="4">
  <BehaviorTree ID="SmokeTest">
    <Sequence>
      <Action ID="DoSomething"/>
      <Action ID="DoSomethingElse"/>
    </Sequence>
  </BehaviorTree>
</root>
EOF

if "$OUT" "$SMOKE_XML" >/dev/null 2>&1; then
  echo "Smoke test PASSED"
else
  echo "Smoke test FAILED — loader built but does not run correctly" >&2
  exit 2
fi
