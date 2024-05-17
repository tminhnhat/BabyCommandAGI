#!/bin/bash
set -ex

# 現在の時刻を取得してフォルダ名に使用
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# バックアップ先のディレクトリを作成
BACKUP_DIR="./workspace_backup/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

# .gitkeep以外のファイルをコピー
# システムタイプの判定
if [ "$(uname)" = "Darwin" ] || [ "$(uname -s)" = "FreeBSD" ]; then
    # BSDベースのシステム（macOSやFreeBSDなど）
    echo "BSD-based system detected. Using rsync for backup."
    rsync -av --exclude='.gitkeep' ./workspace/ "$BACKUP_DIR/"
else
    # GNU/Linuxベースのシステム
    echo "GNU/Linux system detected. Using find and cp for backup."
    find ./workspace -type f ! -name ".gitkeep" -exec cp --parents \{\} "$BACKUP_DIR" \;
fi

echo "Backup completed successfully to $BACKUP_DIR"
