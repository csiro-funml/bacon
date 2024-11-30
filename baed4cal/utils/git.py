import subprocess

def get_git_revision_hash() -> str:
    """Get git revision hash identifying the current state of the local repository.

    Subprocess will be run calling git. If that fails, an exception will be raised.

    NOTE: Originally from: https://stackoverflow.com/a/21901260

    Returns:
        str: git revision hash
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    """Get git revision short hash identifying the current state of the local repository.

    Subprocess will be run calling git. If that fails, an exception will be raised.

    NOTE: Originally from: https://stackoverflow.com/a/21901260

    Returns:
        str: git revision short hash
    """
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
