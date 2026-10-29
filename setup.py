"""Include the canonical root configuration and frozen Wiki in built wheels."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithAssets(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / 'src' / 'assets'
        for directory in ('configs', 'knowledge/kernelwiki_public'):
            for source in (root / directory).rglob('*'):
                if not source.is_file() or '__pycache__' in source.parts:
                    continue
                relative = source.relative_to(root)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)


setup(cmdclass={'build_py': BuildWithAssets})
