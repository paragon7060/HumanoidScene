"""Save terminal success states before auto-reset; no full rollout in RAM."""

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTermCfg, DatasetExportMode
from isaaclab.utils import configclass
from ..mdp.reset_bank import SuccessSnapshotRecorder


@configclass
class RecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    success_states = RecorderTermCfg(class_type=SuccessSnapshotRecorder)
