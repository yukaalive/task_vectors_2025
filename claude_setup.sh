# システムパッケージとNode.js環境のセットアップ
apt-get update
apt-get install -y curl git

# NVMのインストール
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# NVMの読み込み（新しいシェルセッションで必要）
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Node.jsのインストールと使用
nvm install 23.5.0
nvm use 23.5.0
node -v

# Claude Codeのインストール
npm install -g @anthropic-ai/claude-code