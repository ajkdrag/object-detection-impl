from enum import Enum

from visionlab.datamodules.parsers.cls_subfolder import ClassSubfolderParser
from visionlab.datamodules.parsers.splits_subfolder import (
    SplitsSubfolderParser,
)


class FolderStructs(Enum):
    CLS_SUBFOLDER = ClassSubfolderParser
    TRAIN_VAL_TEST = SplitsSubfolderParser
