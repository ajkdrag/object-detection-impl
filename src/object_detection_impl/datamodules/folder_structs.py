from enum import Enum

from object_detection_impl.datamodules.parsers.cls_subfolder import ClassSubfolderParser
from object_detection_impl.datamodules.parsers.splits_subfolder import (
    SplitsSubfolderParser,
)


class FolderStructs(Enum):
    CLS_SUBFOLDER = ClassSubfolderParser
    TRAIN_VAL_TEST = SplitsSubfolderParser
