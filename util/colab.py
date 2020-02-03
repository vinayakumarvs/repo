from ..imports import *

def mount_google_drive(gdrive_path: str = '../content/drive/'):
    """
    Mount Google Drive. Do nothing if Google Drive is already mounted.
    :param gdrive_path: local path to mount Google Drive to.
    :return: None.
    """
    drive.mount (gdrive_path, force_remount=True)