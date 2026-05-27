#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  JANUS Telepad — ship the portable package to a remote GPU
# ─────────────────────────────────────────────────────────────

CYAN='\033[96m'
GREEN='\033[92m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARBALL="$SCRIPT_DIR/../portable.tar.gz"
KEY_PATH="$HOME/.ssh/janus_telepad"

echo -e "${BOLD}═══════════════════════════════════════${RESET}"
echo -e "${BOLD}  JANUS Telepad${RESET}"
echo -e "${DIM}  Ship portable training to remote GPU${RESET}"
echo -e "${BOLD}═══════════════════════════════════════${RESET}"
echo

# ─── Prompt for remote info ───────────────────────────────────

read -rp "  Remote IP address: " REMOTE_IP
read -rp "  Remote user [root]: " REMOTE_USER
REMOTE_USER="${REMOTE_USER:-root}"
read -rp "  Remote port [22]: " REMOTE_PORT
REMOTE_PORT="${REMOTE_PORT:-22}"

echo

# ─── Generate SSH key if needed ───────────────────────────────

if [ ! -f "$KEY_PATH" ]; then
    echo -e "${CYAN}Generating SSH key...${RESET}"
    ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "janus-telepad"
    echo
    echo -e "${GREEN}Key generated:${RESET} $KEY_PATH"
    echo
    echo -e "${BOLD}Copy this public key to your remote machine:${RESET}"
    echo -e "${DIM}─────────────────────────────────────────────${RESET}"
    cat "${KEY_PATH}.pub"
    echo -e "${DIM}─────────────────────────────────────────────${RESET}"
    echo
    echo -e "  Paste it into ${BOLD}~/.ssh/authorized_keys${RESET} on the remote,"
    echo -e "  or run this from your local machine:"
    echo
    echo -e "  ${CYAN}ssh-copy-id -i ${KEY_PATH}.pub -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_IP}${RESET}"
    echo
    read -rp "  Press Enter once the key is installed on the remote... "
    echo
else
    echo -e "${GREEN}Using existing key:${RESET} $KEY_PATH"
    echo
fi

SSH_CMD="ssh -i $KEY_PATH -p $REMOTE_PORT ${REMOTE_USER}@${REMOTE_IP}"
SCP_CMD="scp -i $KEY_PATH -P $REMOTE_PORT"

# ─── Test connection ──────────────────────────────────────────

echo -e "${CYAN}Testing connection...${RESET}"
if $SSH_CMD "echo ok" 2>/dev/null; then
    echo -e "${GREEN}Connected.${RESET}"
else
    echo -e "\033[91mConnection failed.${RESET} Check IP, user, port, and key."
    echo
    echo -e "  Manual connect command:"
    echo -e "  ${CYAN}${SSH_CMD}${RESET}"
    exit 1
fi
echo

# ─── Build tarball if needed ──────────────────────────────────

if [ ! -f "$TARBALL" ]; then
    echo -e "${CYAN}Packing portable tarball...${RESET}"
    cd "$SCRIPT_DIR/.."
    tar czf portable.tar.gz portable/
    TARBALL="$SCRIPT_DIR/../portable.tar.gz"
fi
echo -e "  tarball: $(du -sh "$TARBALL" | cut -f1)"

# ─── Ship it ──────────────────────────────────────────────────

echo -e "${CYAN}Uploading...${RESET}"
$SCP_CMD "$TARBALL" "${REMOTE_USER}@${REMOTE_IP}:~/portable.tar.gz"
echo -e "${GREEN}Uploaded.${RESET}"
echo

# ─── Unpack + install deps on remote ─────────────────────────

echo -e "${CYAN}Unpacking and installing dependencies on remote...${RESET}"
$SSH_CMD bash <<'REMOTE_SETUP'
set -e
cd ~
tar xzf portable.tar.gz
cd portable
pip install numpy sentence-transformers torch 2>/dev/null \
  || pip3 install numpy sentence-transformers torch
echo "READY"
REMOTE_SETUP

echo
echo -e "${GREEN}Remote is ready.${RESET}"
echo

# ─── Print launch commands ────────────────────────────────────

echo -e "${BOLD}═══════════════════════════════════════${RESET}"
echo -e "${BOLD}  Commands${RESET}"
echo -e "${BOLD}═══════════════════════════════════════${RESET}"
echo
echo -e "  ${DIM}# Connect to remote:${RESET}"
echo -e "  ${CYAN}${SSH_CMD}${RESET}"
echo
echo -e "  ${DIM}# Run training (full corpus):${RESET}"
echo -e "  ${CYAN}${SSH_CMD} 'cd ~/portable && python3 train_remote.py'${RESET}"
echo
echo -e "  ${DIM}# Run training in background (detached):${RESET}"
echo -e "  ${CYAN}${SSH_CMD} 'cd ~/portable && nohup python3 train_remote.py > train.log 2>&1 &'${RESET}"
echo
echo -e "  ${DIM}# Watch training log:${RESET}"
echo -e "  ${CYAN}${SSH_CMD} 'tail -f ~/portable/train.log'${RESET}"
echo
echo -e "  ${DIM}# Pull weights home when done:${RESET}"
echo -e "  ${CYAN}${SCP_CMD} ${REMOTE_USER}@${REMOTE_IP}:~/portable/link_mlp_trained.json $SCRIPT_DIR/../../link_mlp_trained.json${RESET}"
echo
echo -e "  ${DIM}# Pull cache home (preserves embeddings for future runs):${RESET}"
echo -e "  ${CYAN}${SCP_CMD} ${REMOTE_USER}@${REMOTE_IP}:~/portable/data/embedding_cache.jsonl $SCRIPT_DIR/embedding_cache.jsonl${RESET}"
echo
