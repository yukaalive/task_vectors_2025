#!/bin/bash
# NVMを現在のセッションで読み込む
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Node.js 23.5.0のインストール
nvm install 23.5.0

# Node.js 23.5.0を使用
nvm use 23.5.0

# バージョン確認
node -v

# Claude Codeのインストール
npm install -g @anthropic-ai/claude-code
