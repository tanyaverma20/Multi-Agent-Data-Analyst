# src/tools/agent_tools.py
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) == "src" else BASE_DIR
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from tools.file_tools import FileTools
    from tools.dataset_tools import DatasetTools
    from tools.model_tools import ModelTools
    from tools.notebook_tools import NotebookTools
    from tools.job_tools import JobTools
    from tools.memory_tools import MemoryTools
    from tools.feature_tools import FeatureTools
except ImportError:
    from src.tools.file_tools import FileTools
    from src.tools.dataset_tools import DatasetTools
    from src.tools.model_tools import ModelTools
    from src.tools.notebook_tools import NotebookTools
    from src.tools.job_tools import JobTools
    from src.tools.memory_tools import MemoryTools
    from src.tools.feature_tools import FeatureTools


class ToolRegistry:
    """Central registry for all internal MCP-style tools."""

    def __init__(self):
        self.tools = {
            "file": FileTools(),
            "dataset": DatasetTools(),
            "model": ModelTools(),
            "notebook": NotebookTools(),
            "job": JobTools(),
            "memory": MemoryTools(),
            "feature": FeatureTools(),
        }

    def get_tool(self, name: str):
        """Get a tool instance by name."""
        return self.tools.get(name, None)

    def list_tools(self):
        """List all tool names."""
        return list(self.tools.keys())
