#!/bin/bash
set -euo pipefail
cd /data2/cmdir/home/toandq/CSG-Skin-paperB
MSG=${1:?msg}; shift
[ "$#" -ge 1 ] || { echo "need paths"; exit 1; }
git add "$@"
if git diff --cached --quiet; then
  echo "nothing to commit"
  exit 0
fi
TREE=$(git write-tree)
COMMIT=$(GIT_AUTHOR_NAME='vinhtranduc' GIT_AUTHOR_EMAIL='vinhtran9969@gmail.com' \
  GIT_COMMITTER_NAME='vinhtranduc' GIT_COMMITTER_EMAIL='vinhtran9969@gmail.com' \
  git commit-tree "$TREE" -p HEAD -m "$MSG")
git reset --hard "$COMMIT"
git push origin HEAD:paperb/writing-r6
git log -1 --format='%H %an <%ae>%n%s'
