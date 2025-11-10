import os
from google.cloud import storage
from tqdm import tqdm

def load_from_gcs(gcs_path, local_path):
    """
    Downloads a file or all files in a directory from a GCP bucket to a local path.

    Args:
        gcs_path (str): Path to a file or directory in the GCP bucket. gs://{bucket_name}/{path}
        local_path (str): Path to the local file or directory where data will be saved.
    """
    gcs_path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = gcs_path_parts[0]
    blob_path = gcs_path_parts[1]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Check if the given GCP path is a file or directory
    blobs = list(bucket.list_blobs(prefix=blob_path))

    if not blobs:
        raise ValueError(f"No files found at {blob_path} in bucket {bucket_name}")

    if len(blobs) == 1 and blobs[0].name == blob_path:  # Single file case
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blobs[0].download_to_filename(local_path)
        print(f"Downloaded {blob_path} to {local_path}")
    else:  # Directory case
        if not local_path.endswith('/'):
            local_path += '/'  # Ensure local directory structure
        os.makedirs(local_path, exist_ok=True)

        for blob in blobs:
            if not blob.name.endswith('/'):  # Ignore "directory" markers
                relative_path = blob.name[len(blob_path):].lstrip('/')  # Remove the prefix
                local_file_path = os.path.join(local_path, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                blob.download_to_filename(local_file_path)
                print(f"Downloaded {blob.name} to {local_file_path}")

    print("Download complete.")
    return local_path

def load_ckpt_from_gcs(checkpoint_path, local_path='/tmp/model.ckpt'):
    ckpt_type, ckpt_path = checkpoint_path.split('::')
    local_path = load_from_gcs(ckpt_path, local_path)
    print(f"Checkpoint downloaded to {local_path}")
    return f'{ckpt_type}::/{local_path}'

def upload_to_gcs(local_path, gcs_path):
    """Uploads a file to Google Cloud Storage."""
    client = storage.Client()

    # Extract bucket name and blob path
    gcs_path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = gcs_path_parts[0]
    blob_path = gcs_path_parts[1]

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"File uploaded to {gcs_path}")


def read_from_gcs(gcs_path):
    """Reads a text file from GCS and returns its content as a string."""
    client = storage.Client()

    # Extract bucket name and blob path
    bucket_name, blob_path = gcs_path.replace("gs://", "").split("/", 1)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"GCS file not found: {gcs_path}")

    return blob.download_as_text().strip()

def gcs_path_exists(gcs_path):
    """Check if a file or directory exists in Google Cloud Storage."""
    client = storage.Client()

    # Extract bucket name and prefix
    bucket_name, prefix = gcs_path.replace("gs://", "").split("/", 1)

    bucket = client.bucket(bucket_name)

    # List objects with the given prefix (works for both files & "directories")
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    return len(blobs) > 0  # Returns True if any object exists

def load_first_n_files_from_gcs(gcs_path, local_path, num_to_download=50):
    """
    Downloads the first `num_to_download` files from a GCP bucket directory to a local path.

    Args:
        gcs_path (str): Path to a directory in the GCP bucket. Format: gs://{bucket_name}/{path}/
        local_path (str): Local directory where the files will be saved.
        num_to_download (int): Number of files to download. Default is 50.

    Returns:
        str: Local path where files are downloaded.
    """
    gcs_path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = gcs_path_parts[0]
    blob_prefix = gcs_path_parts[1].rstrip("/") + "/"  # Ensure it's treated as a directory

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # List all files in the directory
    blobs = list(bucket.list_blobs(prefix=blob_prefix))

    if not blobs:
        raise ValueError(f"No files found at {blob_prefix} in bucket {bucket_name}")

    # Filter out directory markers if they exist
    blobs = [blob for blob in blobs if not blob.name.endswith("/")]

    # Limit to the first `num_to_download` files
    blobs = blobs[:num_to_download]

    # Ensure local directory exists
    os.makedirs(local_path, exist_ok=True)

    for blob in tqdm(blobs, desc=f"Downloading {num_to_download} files"):
        relative_path = blob.name[len(blob_prefix):].lstrip('/')  # Remove the prefix
        local_file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

    print("Download complete.")
    return local_path

from google.cloud import storage
import json
import os

def modify_dataset_info_gcs(gcs_path, local_dir, num_files_to_keep=50):
    """
    Downloads a dataset_info.json file from GCS, modifies it to retain only a subset of files,
    updates the dataset size, and saves the modified file locally.

    Args:
        gcs_path (str): GCS path to dataset_info.json (e.g., gs://bucket/path/dataset_info.json)
        local_dir (str): Local directory to save the modified dataset_info.json
        num_files_to_keep (int): Number of files to retain in the dataset

    Returns:
        str: Path to the modified dataset_info.json file
    """
    # Parse GCS path
    gcs_path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = gcs_path_parts[0]
    blob_path = gcs_path_parts[1]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = os.path.join(local_dir, "dataset_info-full.json")

    # Download dataset_info.json
    blob.download_to_filename(local_file_path)
    print(f"Downloaded dataset_info.json to {local_file_path}")

    # Load JSON file
    with open(local_file_path, "r") as f:
        data = json.load(f)

    # Modify the "download_checksums" field
    if "download_checksums" in data:
        original_files = list(data["download_checksums"].items())
        subset_files = dict(original_files[:num_files_to_keep])
        data["download_checksums"] = subset_files
        print(f"Kept {len(subset_files)} files in 'download_checksums'")

        # Update dataset_size and download_size fields
        new_dataset_size = sum(file_info["num_bytes"] for file_info in subset_files.values())
        data["dataset_size"] = new_dataset_size
        data["download_size"] = new_dataset_size  # Assuming it's the same as dataset_size

    # Modify "splits" field if present
    if "splits" in data and "train" in data["splits"]:
        train_split = data["splits"]["train"]
        if "shard_lengths" in train_split:
            train_split["shard_lengths"] = train_split["shard_lengths"][:num_files_to_keep]
        train_split["num_bytes"] = new_dataset_size
        train_split["num_examples"] = sum(train_split["shard_lengths"])  # Update example count

    # Save the modified file
    modified_file_path = os.path.join(local_dir, "dataset_info.json")
    with open(modified_file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Modified dataset_info.json saved to {modified_file_path}")
    return modified_file_path

def modify_state_json_gcs(gcs_path, local_dir, num_files_to_keep=50):
    """
    Downloads a state.json file from GCS, modifies it to retain only a subset of data files,
    and saves the modified file locally.

    Args:
        gcs_path (str): GCS path to state.json (e.g., gs://bucket/path/state.json)
        local_dir (str): Local directory to save the modified state.json
        num_files_to_keep (int): Number of files to retain in the dataset

    Returns:
        str: Path to the modified state.json file
    """
    # Parse GCS path
    gcs_path_parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name = gcs_path_parts[0]
    blob_path = gcs_path_parts[1]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = os.path.join(local_dir, "state-full.json")

    # Download state.json
    blob.download_to_filename(local_file_path)
    print(f"Downloaded state.json to {local_file_path}")

    # Load JSON file
    with open(local_file_path, "r") as f:
        data = json.load(f)

    # Modify the "_data_files" field
    if "_data_files" in data:
        original_files = data["_data_files"]
        subset_files = original_files[:num_files_to_keep]
        data["_data_files"] = subset_files
        print(f"Kept {len(subset_files)} files in '_data_files'")

    # Save the modified file
    modified_file_path = os.path.join(local_dir, "state.json")
    with open(modified_file_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Modified state.json saved to {modified_file_path}")
    return modified_file_path
