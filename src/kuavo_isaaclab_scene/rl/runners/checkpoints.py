"""Publish a checkpoint filename only after serialization has completed."""

from pathlib import Path
from uuid import uuid4


class AtomicCheckpointMixin:
    def save(self, path, infos=None):
        destination = Path(path)
        pending = destination.with_name(f".{destination.name}.{uuid4().hex}.pending")
        try:
            super().save(str(pending), infos)
            pending.replace(destination)
        finally:
            pending.unlink(missing_ok=True)
