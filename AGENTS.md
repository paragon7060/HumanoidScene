# Shared project context

## Google Drive storage for RL runs

For training, checkpoint storage, or disk-space work, read
[docs/RL_GOOGLE_DRIVE.md](docs/RL_GOOGLE_DRIVE.md) first. It records the existing
connection, verified backup, commands, and limitations; do not infer that a
background uploader or training job is running from the presence of this setup.

On this server, the authenticated rclone remote is **`seonho:`** and the dedicated
destination is **`seonho:HumanoidScene-RL`**. It was authenticated and tested with
real uploads on 2026-09-08. Check current quota/status when needed; recorded
capacity is only a snapshot. Reuse this connection rather than starting a new
login while it works. Do not change Drive sharing or copy credentials to Git,
chat, another account, or another checkout.

The existing checkout provides shared entrypoints for other tasks on this server:

```bash
bash /home/seonho/HumanoidScene/scripts/rl/gdrive.sh about seonho:
python3 /home/seonho/HumanoidScene/scripts/rl/drive_backup.py \
  --run-dir /absolute/path/to/unique-run-directory \
  --remote-root seonho:HumanoidScene-RL --watch 300 --keep 2
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
