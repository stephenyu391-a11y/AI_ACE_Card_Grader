#!/bin/bash
# Publish the project to GitHub: only the files listed in ops/publish/files.txt, after a secret
# and size check. Pictures, models, run outputs and keys can never be published by this script.
#
#   ops/publish_github.sh --check              # build and check, show what would change; nothing is pushed
#   ops/publish_github.sh "message"            # build, check, commit and push
#
# Once: a deploy key for this repository (made by the first run) must be added on GitHub:
#   repository -> Settings -> Deploy keys -> Add deploy key, paste ~/.ssh/github_ace_grader.pub,
#   tick "Allow write access".
set -euo pipefail
source "$(dirname "$0")/config.sh"
REPO_SSH="git@github.com:stephenyu391-a11y/AI_ACE_Card_Grader.git"
AUTHOR_NAME="stephenyu391-a11y"
AUTHOR_EMAIL="stephenyu391-a11y@users.noreply.github.com"   # GitHub's private address: keeps your e-mail out of the history
DEPLOY_KEY="$HOME/.ssh/github_ace_grader"
STAGE="$ROOT/.publish"                                       # a local clone of the repository
MAX_FILE_MB=50                                               # GitHub warns at 50 MB and refuses 100 MB
export GIT_SSH_COMMAND="ssh -i $DEPLOY_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

[ -f "$DEPLOY_KEY" ] || { ssh-keygen -q -t ed25519 -N '' -C ai-ace-card-grader-deploy -f "$DEPLOY_KEY"
    echo "new deploy key; add this on GitHub (Settings -> Deploy keys, allow write access):"; cat "$DEPLOY_KEY.pub"; }

# 1. A clean copy of the repository.
if [ ! -d "$STAGE/.git" ]; then
    git init -q -b main "$STAGE"
    git -C "$STAGE" remote add origin "$REPO_SSH"
    git -C "$STAGE" fetch -q origin 2>/dev/null && git -C "$STAGE" rev-parse -q --verify origin/main >/dev/null \
        && git -C "$STAGE" reset -q --hard origin/main || true
fi
git -C "$STAGE" config user.name "$AUTHOR_NAME"
git -C "$STAGE" config user.email "$AUTHOR_EMAIL"
git -C "$STAGE" ls-files -z | (cd "$STAGE" && xargs -0r rm -f)   # files dropped from the list disappear too

# 2. Copy exactly the listed files.
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    src="${line%% -> *}"; dst="${line##* -> }"
    [ -f "$ROOT/$src" ] || { echo "missing: $src"; exit 1; }
    mkdir -p "$STAGE/$(dirname "$dst")"
    cp "$ROOT/$src" "$STAGE/$dst"
done < "$ROOT/ops/publish/files.txt"
find "$STAGE" -mindepth 1 -type d -empty -not -path "$STAGE/.git*" -delete

# 3. Checks: no secrets, no large files.
cd "$STAGE"
git add -A
secrets='BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|[0-9a-f]{64}'
if [ -s "$HOME/.config/vastai/vast_api_key" ]; then secrets="$secrets|$(cat "$HOME/.config/vastai/vast_api_key")"; fi
if git grep -qEI --cached "$secrets"; then
    echo "STOPPED: something that looks like a key or token is in these files:"; git grep -lEI --cached "$secrets"; exit 1
fi
big=$(git ls-files -z | xargs -0 du -m | awk -v max=$MAX_FILE_MB '$1 >= max')
[ -z "$big" ] || { echo "STOPPED: files of $MAX_FILE_MB MB or more:"; echo "$big"; exit 1; }
echo "checks passed: $(git ls-files | wc -l) files, $(git ls-files -z | xargs -0 du -ch | tail -1 | cut -f1) in all"
git status --short

# 4. Commit and push.
[ "${1:-}" = --check ] && { echo "check only: nothing committed or pushed"; exit 0; }
message="${1:?give a commit message, or --check}"
git diff --cached --quiet && { echo "nothing changed since the last publish"; exit 0; }
git commit -q -m "$message"
git push -u origin main
echo "published: https://github.com/stephenyu391-a11y/AI_ACE_Card_Grader"
