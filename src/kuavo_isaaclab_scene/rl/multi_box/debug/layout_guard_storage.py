"""Store whole-wave reset evidence once, with explicit outcome references."""
import json
import re

FORMAT='whole_wave_initial_layout_guard_reference_v1'


def store_layout_guard(directory,wave,environment_count,guard):
    if type(wave) is not int or wave<0 or type(environment_count) is not int or environment_count<1:
        raise ValueError('Expected nonnegative wave and positive environment count')
    filename=f'initial_layout_guard_wave_{wave:04d}.json'
    with (directory/filename).open('x') as stream:
        json.dump(dict(format=FORMAT,wave=wave,environment_count=environment_count,guard=guard),stream)
        stream.write('\n')
    return [dict(storage=FORMAT,file=filename,wave=wave,environment=i) for i in range(environment_count)]


def load_layout_guard(outcome,directory):
    """Accept historical inline guards and verify references from new runs."""
    reference=outcome['initial_layout_guard']
    if reference.get('storage')!=FORMAT:return reference
    filename=reference['file']
    if not isinstance(filename,str) or not re.fullmatch(r'initial_layout_guard_wave_\d{4,}\.json',filename):
        raise ValueError('Layout guard reference must name a local canonical wave file')
    path=directory/filename
    if path.is_symlink():raise ValueError('Layout guard reference cannot be a symlink')
    packet=json.loads(path.read_text())
    if packet.get('format')!=FORMAT or packet.get('wave')!=outcome['wave'] \
            or reference['wave']!=outcome['wave'] or reference['environment']!=outcome['environment'] \
            or not 0<=outcome['environment']<packet['environment_count']:
        raise ValueError('Layout guard reference does not match this outcome')
    return packet['guard']
