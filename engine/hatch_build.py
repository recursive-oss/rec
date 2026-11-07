import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        subprocess.run([sys.executable, str(Path(self.root) / "native/build.py")], check=True)
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
        build_data["artifacts"] = [
            "src/rec_mlx/_grammar_native" + sysconfig.get_config_var("EXT_SUFFIX")
        ]
