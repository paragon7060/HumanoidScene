"""Keep CUDA and graphics on one GPU without changing host device access.

CUDA_VISIBLE_DEVICES alone does not filter Vulkan/GL enumeration. A private
mount namespace exposes just the selected NVIDIA device. UUID selection keeps
CUDA correct when the visible device numbering changes inside the namespace.
No driver, permission, credential, or other process changes are performed.
"""
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

MARKER = 'HUMANOIDSCENE_SINGLE_GPU_NAMESPACE'


def validate_namespace(record, gpu, *, devices, cuda_mask):
    expected = '/dev/nvidia'+str(record['minor_number'])
    if record['physical_gpu'] != gpu or record['renderer_gpu'] != 0 \
            or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', record['uuid']) \
            or cuda_mask != record['uuid'] or sorted(devices) != [expected]:
        raise ValueError('Single GPU namespace identity or CUDA mask changed')
    return dict(record, CUDA_VISIBLE_DEVICES=cuda_mask,
        graphics_device_access='selected_device_only_private_mount_namespace')


def namespace_command(record, executable, arguments, *, device_exists=None):
    exists = device_exists or Path.exists
    command = ['bwrap', '--bind', '/', '/', '--dev', '/dev']
    for path in ('/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools',
                 '/dev/nvidia'+str(record['minor_number'])):
        if not exists(Path(path)):
            raise ValueError('Required NVIDIA device unavailable: '+path)
        command.extend(('--dev-bind', path, path))
    command.extend(('--bind', '/dev/shm', '/dev/shm'))
    # Keep only this GPU's DRM render node on hosts which provide one.
    for path in record.get('drm_render_nodes', []):
        command.extend(('--dir', '/dev/dri', '--dev-bind', path, path))
    command.extend(('--setenv', 'CUDA_VISIBLE_DEVICES', record['uuid'],
        '--setenv', MARKER, json.dumps(record, separators=(',', ':')),
        '--', executable, *arguments))
    return command


def ensure_single_gpu_namespace(gpu):
    """Re-exec this supervisor before starting any GPU runtime; fail closed."""
    if MARKER in os.environ:
        record = json.loads(os.environ[MARKER])
        return validate_namespace(record, gpu,
            devices=[str(p) for p in Path('/dev').glob('nvidia[0-9]*')],
            cuda_mask=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if sys.platform != 'linux' or shutil.which('bwrap') is None:
        raise RuntimeError('Graphics isolation requires Linux bubblewrap; no fallback to other GPUs')
    row = subprocess.check_output(['nvidia-smi', '--id='+str(gpu),
        '--query-gpu=uuid,minor_number,pci.bus_id', '--format=csv,noheader,nounits'], text=True).strip()
    uuid, minor, bus = [v.strip() for v in row.split(',')]
    record = dict(physical_gpu=gpu, uuid=uuid, minor_number=int(minor),
        PCI_bus_id=bus, renderer_gpu=0, drm_render_nodes=[])
    device = Path('/dev/nvidia'+minor)
    if not stat.S_ISCHR(device.stat().st_mode):
        raise RuntimeError('Selected NVIDIA device is not a character device')
    pci = bus.lower().removeprefix('00000000:')
    for node in Path('/sys/class/drm').glob('renderD*'):
        if (node/'device').resolve().name.lower().endswith(pci):
            record['drm_render_nodes'].append('/dev/dri/'+node.name)
    # Validate before executing; GPU metadata stays host-local.
    validate_namespace(record, gpu, devices=[str(device)], cuda_mask=uuid)
    command = namespace_command(record, sys.executable, sys.argv)
    os.execvp(command[0], command)
    raise AssertionError('exec must not return')
