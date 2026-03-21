#!/usr/bin/env python
"""Setup script for pymgcv."""

from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
	"""Build native extensions when toolchains are available.

	If compilation fails (e.g., missing compiler), installation still
	succeeds and pymgcv falls back to the pure-Python path.
	"""

	def run(self) -> None:
		try:
			super().run()
		except Exception as exc:  # pragma: no cover
			print(f"WARNING: native extensions were skipped: {exc}")

	def build_extension(self, ext: Extension) -> None:
		try:
			super().build_extension(ext)
		except Exception as exc:  # pragma: no cover
			print(f"WARNING: failed to build extension {ext.name}: {exc}")


def _get_extensions() -> list[Extension]:
	import numpy

	return [
		Extension(
			"pymgcv.linalg._native_c",
			sources=["pymgcv/linalg/_native_c.c"],
			include_dirs=[numpy.get_include()],
		)
	]


setup(
	ext_modules=_get_extensions(),
	cmdclass={"build_ext": OptionalBuildExt},
)
