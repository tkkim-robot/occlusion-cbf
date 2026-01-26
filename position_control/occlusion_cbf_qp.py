"""
Occlusion CBF - Extension of Backup CBF for Occlusion Safety
"""

from safe_control.position_control.backup_cbf_qp import BackupCBF

class OcclusionCBF(BackupCBF):
    """
    Occlusion Control Barrier Function.
    
    Currently inherits all functionality from BackupCBF.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
