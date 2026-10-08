# src/tools/memory_tools.py
import os
import json

class MemoryTools:
    """
    Simple JSON-based key-value storage for agent memory.
    Supports both load(key) and load() for full snapshot compatibility.
    """

    def __init__(self, storage_dir="streamlit_app_storage/memory"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.storage_file = os.path.join(self.storage_dir, "memory.json")

        # If memory file missing or corrupted → reset
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, "w") as f:
                json.dump({}, f)
        else:
            try:
                with open(self.storage_file, "r") as f:
                    json.load(f)
            except Exception:
                with open(self.storage_file, "w") as f:
                    json.dump({}, f)

    # ---------------------------------------------------------
    # SAVE KEY
    # ---------------------------------------------------------
    def save(self, key, value):
        try:
            with open(self.storage_file, "r") as f:
                data = json.load(f)

            data[key] = value

            with open(self.storage_file, "w") as f:
                json.dump(data, f, indent=4)

            return {"status": "success", "saved_key": key}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ---------------------------------------------------------
    # LOAD SPECIFIC KEY OR ALL
    # ---------------------------------------------------------
    def load(self, key=None):
        try:
            with open(self.storage_file, "r") as f:
                data = json.load(f)

            if key is None:
                return {"status": "success", "memory": data}

            return data.get(key, None)

        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ---------------------------------------------------------
    # LOAD ALL MEMORY
    # ---------------------------------------------------------
    def load_all(self):
        try:
            with open(self.storage_file, "r") as f:
                data = json.load(f)

            return data

        except Exception as e:
            return {"status": "error", "error": str(e)}
