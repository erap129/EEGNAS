from utilities.report_generation import concat_and_pivot_results, get_base_folder_name
from os import listdir
from os.path import isfile, join
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import os


def connect_to_gdrive():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)
    return drive


def upload_exp_to_gdrive(fold_names, first_dataset):
    base_folder_name = get_base_folder_name(fold_names, first_dataset)
    drive = connect_to_gdrive()
    base_folder = drive.CreateFile({'title': base_folder_name,
                                   'parents': [{"id": '1z6y-g4HqmQm7i8R2h66sDd5e6AV1IhVM'}],
                                    'mimeType': "application/vnd.google-apps.folder"})
    base_folder.Upload()
    concat_filename = concat_and_pivot_results(fold_names, first_dataset)
    file_drive = drive.CreateFile({'title': concat_filename,
                                   'parents': [{"id": base_folder['id']}]})
    file_drive.SetContentFile(concat_filename)
    file_drive.Upload()
    os.remove(concat_filename)
    for folder in fold_names:
        full_folder = 'results/' + folder
        if os.path.isdir(full_folder):
            spec_folder = drive.CreateFile({'title': folder,
                                            'parents': [{"id": base_folder['id']}],
                                            'mimeType': "application/vnd.google-apps.folder"})
            spec_folder.Upload()
            files = [f for f in listdir(full_folder) if isfile(join(full_folder, f))]
            for filename in files:
                if '.p' not in filename:
                    file_drive = drive.CreateFile({'title': filename,
                                                       'parents': [{"id": spec_folder['id']}]})
                    file_drive.SetContentFile(str(join(full_folder, filename)))
                    file_drive.Upload()
