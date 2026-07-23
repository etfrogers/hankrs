#!/usr/bin/env bash
set -euo pipefail

# Benchmark a range of Git commits using Criterion and critcmp.
# Usage: ./bench_commits.sh [git-commit-range]
# Example: ./bench_commits.sh main..HEAD

OLD_COMMIT="${1:-main}"
NEW_COMMIT="${2:-HEAD}"

# 1. Ensure working directory is clean before switching commits
if [[ -n $(git status --porcelain) ]]; then
  echo "Error: Working directory has uncommitted changes. Stash or commit them first." >&2
  exit 1
fi

# 2. Store original position and set up cleanup trap (triggers on exit or Ctrl+C)
ORIGINAL_HEAD=$(git rev-parse --abbrev-ref HEAD)
if [[ "$ORIGINAL_HEAD" == "HEAD" ]]; then
  ORIGINAL_HEAD=$(git rev-parse HEAD)
fi

cleanup() {
  echo -e "\n[!] Restoring repository to $ORIGINAL_HEAD..."
  git checkout -q "$ORIGINAL_HEAD"
}
trap cleanup EXIT

OLD_HASH=$(git rev-parse "$OLD_COMMIT")
AFTER_OLD_COMMITS=$(git rev-list --reverse "$OLD_COMMIT..$NEW_COMMIT")
# echo "Found $(echo "$AFTER_OLD_COMMITS" | wc -l) commits after $OLD_COMMIT."

COMMITS=("$OLD_HASH")
for c in $AFTER_OLD_COMMITS; do
    # echo "  $c"
  COMMITS+=("$c")
done
# echo "${COMMITS[@]}"

if [[ -z "$COMMITS" ]]; then
  echo "Error: No commits found in range '$OLD_COMMIT..$NEW_COMMIT'." >&2
  exit 1
fi

BASELINES=()

echo "Starting benchmark sweep across ${#COMMITS[@]} commits: $OLD_COMMIT..$NEW_COMMIT"
echo "================================================"

# 4. Iterate over each commit
for COMMIT in "${COMMITS[@]}"; do
  # echo "$COMMIT"
  SHORT_HASH=$(git rev-parse --short "$COMMIT")

  echo -e "\n--> Checking out [$SHORT_HASH]: $(git log -1 --format='%s' "$COMMIT")"
  git checkout -q "$COMMIT"

  # Run criterion benchmark saving the baseline under the short hash
  cargo bench --bench hankel_benchmark -- --save-baseline "$SHORT_HASH"

  BASELINES+=("$SHORT_HASH")
done

echo "Running last element with blas"
cargo bench --bench hankel_benchmark --features blas -- --save-baseline "$SHORT_HASH-blas"

echo -e "\n================================================"
echo "Benchmarks complete! Running critcmp comparison:"
echo "================================================"

echo "All baselines"
echo "${BASELINES[@]}"
echo "${BASELINES[@]}" > baselines.txt

# 5. Output comparison table
critcmp "${BASELINES[@]}" --list
