"""
AgentAugi integration modules for external services.
"""
from .openclaw_bridge import OpenClawBridge, OpenClawConfig, OpenClawTask, OpenClawResult

__all__ = [
    "OpenClawBridge",
    "OpenClawConfig",
    "OpenClawTask",
    "OpenClawResult",
]
