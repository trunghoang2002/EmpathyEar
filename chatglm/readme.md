# Setup

```bash
conda create -n myenv python=3.8
conda activate myenv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"
pip install -r requirements.txt
```