# Shared project context

## Google Drive storage for RL runs

For training, checkpoint storage, or disk-space work, read
[docs/RL_GOOGLE_DRIVE.md](docs/RL_GOOGLE_DRIVE.md) first. It records the existing
connection, verified backup, commands, and limitations; do not infer that a
background uploader or training job is running from the presence of this setup.

The authenticated rclone remote name is host-local configuration. Discover it
with `gdrive.sh listremotes` or set `RL_DRIVE_REMOTE_ROOT`; do not record the
account-specific alias in Git. The dedicated destination is the
`HumanoidScene-RL` folder. Reuse the existing connection rather than starting a
new login while it works. Do not change Drive sharing or copy credentials to
Git, chat, another account, or another checkout.

The existing checkout provides shared entrypoints for other tasks on this server:

```bash
cd /path/to/HumanoidScene
export RL_DRIVE_REMOTE_ROOT='<remote>:HumanoidScene-RL'
bash scripts/rl/gdrive.sh about "${RL_DRIVE_REMOTE_ROOT%%:*}:"
python3 scripts/rl/drive_backup.py \
  --run-dir /absolute/path/to/unique-run-directory \
  --remote-root "$RL_DRIVE_REMOTE_ROOT" --watch 300 --keep 2
```

The wrapper privately uses the ignored `.external/rclone-auth/rclone.conf` in
that checkout; it never needs to be displayed. The Python uploader uses no GPU
or Isaac runtime. Use a distinct run directory name for each experiment because
the remote subfolder is the run directory basename. A manifest is required.
The uploader locks each local run to prevent duplicate backup workers.

This is **upload plus limited local checkpoint retention**, not a mounted disk:
local writes still happen, current logs remain local, and only checksum-verified
older checkpoints are pruned. Keep the newest two and the newest two verified
checkpoints per format. Use `--finished` only after the run's writers have stopped
to include logs. Do not delete active logs or unrelated users' files/processes.
SAC/DPPO's independent checkpoint retention can remove files before this uploader
sees them; read the documented limitation before promising full history.
