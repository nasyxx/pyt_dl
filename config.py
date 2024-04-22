#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     :
email    : Nasy <nasyxx+python@gmail.com>
filename : config.py
project  : pyt_dl
license  : GPL-3.0+

Configurations
"""

from typing import Literal
from equinox import Module, field
from nadl import PG
from rich.console import Console
import tyro


class Optim(Module):
  """Optimizers parameters."""

  op: Literal["lion", "lamb", "adamw", "adamwx"] = "lion"

  lr: float = 1e-3
  b1: float = 0.9
  b2: float = 0.999
  weight_decay: float = 1e-2
  eps: float = 1e-8

  @property
  def name(self) -> str:
    """Get optimizer name."""
    return self.op


class Params(Module):
  """Hyper parameters."""

  seed: int = 7
  batch_size: dict[str, int] = field(default_factory=lambda: {"train": 64, "test": 64})

  epochs: int = 100

  optim: Optim = field(default_factory=Optim)


class Conf(Module):
  """Configurations."""

  pname: str = ""

  _pg: PG = field(init=False, repr=False)
  disable_progress: bool = field(default=False, repr=False)

  def __post_init__(self) -> None:
    """Post initialization."""
    self._pg = PG.init_progress(
      show_progress=not self.disable_progress,
      # extra_columns=
    )

  @property
  def console(self) -> Console:
    """Get console."""
    return self._pg.console

  @property
  def pg(self) -> PG:
    """Get pg."""
    return self._pg

  def toggle_progress(self) -> None:
    """Toggle progress."""
    self.disable_progress = not self.disable_progress
    self.console.log(
      f"Progress is {'enabled' if not self.disable_progress else 'disabled'}"
    )


def build_config_cli() -> type[Conf]:
  """Build config cli."""
  return tyro.extras.subcommand_type_from_defaults({
    "train": Conf(),
    "infer": Conf(disable_progress=True),
  })


if __name__ == "__main__":
  tyro.cli(build_config_cli())
