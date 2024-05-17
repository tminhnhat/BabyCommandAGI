#!/bin/bash
set -ex

# .envファイルのパス
ENV_FILE=".env"

# RESULTS_SOTEE_NUMBERの現在の値を取得
# システムタイプの判定
if [ "$(uname)" = "Darwin" ] || [ "$(uname -s)" = "FreeBSD" ]; then
    CURRENT_VALUE=$(awk -F '=' '/^RESULTS_SOTRE_NUMBER/ {print $2}' "$ENV_FILE")
else
    CURRENT_VALUE=$(grep -oP '(?<=RESULTS_SOTRE_NUMBER=).*' "$ENV_FILE")
fi

# 現在の値をインクリメント
NEW_VALUE=$((CURRENT_VALUE + 1))

# .envファイルのRESULTS_SOTEE_NUMBERを更新
sed -i ".backup" "s/RESULTS_SOTRE_NUMBER=$CURRENT_VALUE/RESULTS_SOTRE_NUMBER=$NEW_VALUE/" "$ENV_FILE"

echo "RESULTS_SOTRE_NUMBER has been incremented to $NEW_VALUE"