import pytest
from single_gpu_runtime import validate_namespace, namespace_command


def record():
    return dict(physical_gpu=3, uuid='GPU-00000000-0000-0000-0000-000000000003',
        minor_number=3, renderer_gpu=0, drm_render_nodes=[])


def test_uuid_mask_and_only_selected_device_are_required():
    r=record()
    assert validate_namespace(r,3,devices=['/dev/nvidia3'],cuda_mask=r['uuid'])['renderer_gpu']==0
    for gpu,devices,mask in ((2,['/dev/nvidia3'],r['uuid']),
            (3,['/dev/nvidia0','/dev/nvidia3'],r['uuid']), (3,['/dev/nvidia3'],'3')):
        with pytest.raises(ValueError):validate_namespace(r,gpu,devices=devices,cuda_mask=mask)


def test_namespace_does_not_expose_other_devices_and_preserves_arguments():
    r=record(); arguments=['script.py','--label','with spaces']
    command=namespace_command(r,'/python',arguments,device_exists=lambda _:True)
    for excluded in ('/dev/nvidia0','/dev/nvidia1','/dev/nvidia2'):
        assert excluded not in command
    assert command[-len(arguments):]==arguments
    assert '--unshare-pid' not in command  # Recorded writer PID remains its host PID.
    assert r['uuid'] in command
    with pytest.raises(ValueError):
        namespace_command(r,'/python',arguments,device_exists=lambda p:str(p)!='/dev/nvidia3')
