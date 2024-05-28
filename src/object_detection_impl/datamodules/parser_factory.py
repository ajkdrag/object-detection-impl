from object_detection_impl.datamodules.folder_structs import FolderStructs


class FolderParserFactory:
    @staticmethod
    def get_folder_parser(folder_struct: str):
        return FolderStructs[folder_struct]
