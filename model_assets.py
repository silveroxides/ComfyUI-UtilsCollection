import os
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote
import folder_paths
from huggingface_hub import hf_hub_download


MODEL_MIGRATIONS = json.loads((Path(__file__).parent / "models" / "migrations.json").read_text(encoding="utf-8"))


def get_model_migration(kind):
    if MODEL_MIGRATIONS.get("version") != 1:
        raise ValueError("Unsupported model migration manifest version")
    try:
        return MODEL_MIGRATIONS["models"][kind]
    except KeyError as error:
        raise ValueError(f"Unknown model migration: {kind}") from error


def download_huggingface_model(category, filename, repo_id, repo_path):
    """Reuse a registered checkpoint or download it on explicit node execution."""
    existing = folder_paths.get_full_path(category, filename)
    if existing and os.path.isfile(existing) and os.path.getsize(existing) > 0:
        return existing
    directories = folder_paths.get_folder_paths(category)
    if not directories:
        raise ValueError(f"No model directory registered for {category}")
    if os.path.basename(filename) != filename:
        raise ValueError("Downloaded model filename must be a plain filename")
    source = hf_hub_download(repo_id=repo_id, filename=repo_path)
    directory = directories[0]
    os.makedirs(directory, exist_ok=True)
    destination = os.path.join(directory, filename)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{filename}.", suffix=".tmp", delete=False) as temporary:
        temporary_path = temporary.name
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return destination


def require_huggingface_model(category, filename, repo_id, repo_path):
    """Return an existing model path or raise an actionable installation error."""
    import folder_paths

    try:
        existing = folder_paths.get_full_path_or_raise(category, filename)
    except Exception:
        existing = None
    if existing and os.path.isfile(existing) and os.path.getsize(existing) > 0:
        return existing

    directories = list(folder_paths.get_folder_paths(category))
    url = f"https://huggingface.co/{repo_id}/blob/main/{quote(repo_path, safe='/')}"
    if not directories:
        raise ValueError(
            f"Required model {filename} was not found, and ComfyUI has no registered "
            f"model directory for {category!r}. Download it from {url}."
        )
    expected = [os.path.abspath(os.path.join(directory, filename)) for directory in directories]
    locations = "\n".join(f"  - {path}" for path in expected)
    raise ValueError(
        f"Required model {filename} was not found. Download it from:\n"
        f"  {url}\n"
        f"Place it in one of ComfyUI's registered {category} directories:\n{locations}"
    )
