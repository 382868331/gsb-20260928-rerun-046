"""Persistent, branch-isolated state storage.

Each branch of a job owns exactly one state file.  Revisions derive a new
branch by copying the state file; the source branch is never mutated, so a
failed revision cannot corrupt the original graph or sibling revisions.
"""

from __future__ import annotations

import copy
import json
import os


class Store:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, branch: str) -> str:
        if not branch or os.sep in branch or branch.startswith("."):
            raise ValueError(f"invalid branch name: {branch!r}")
        return os.path.join(self.root, f"{branch}.json")

    def exists(self, branch: str) -> bool:
        return os.path.exists(self._path(branch))

    def load(self, branch: str) -> dict:
        with open(self._path(branch), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save(self, branch: str, state: dict) -> None:
        path = self._path(branch)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)  # atomic checkpoint

    def derive(self, src_branch: str, dst_branch: str) -> dict:
        """Copy-on-write derivation: dst gets a deep copy, src untouched."""
        state = copy.deepcopy(self.load(src_branch))
        self.save(dst_branch, state)
        return state
