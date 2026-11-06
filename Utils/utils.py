import boto3
import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [reader.pages[i].extract_text() for i in range(len(reader.pages))]
    return "\r\n--- Page Break ---\r\n".join(pages)

def save_to_s3(text: str, prefix: str, object_name: str):
    try:
        s3_client = boto3.client('s3',
                                region_name="us-east-1",
                                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                                aws_session_token=os.environ["AWS_SESSION_TOKEN"])
        s3_client.put_object(Body=text, Bucket=os.environ["BUCKET_NAME"], Key=prefix + '/' + object_name)
    except Exception as e:
        print(f"Error putting object to S3: {e}")

def list_pdfs_in_s3(prefix: str):
    try:
        s3_client = boto3.client('s3',
                                region_name="us-east-1",
                                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                                aws_session_token=os.environ["AWS_SESSION_TOKEN"])
        response = s3_client.list_objects_v2(Bucket=os.environ["BUCKET_NAME"], Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]
    except Exception as e:
        print(f"Error listing objects in S3: {e}")
        return []

def upload_pdfs_to_s3(local_directory: str, s3_prefix: str):
    try:
        s3_client = boto3.client('s3',
                                region_name="us-east-1",
                                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                                aws_session_token=os.environ["AWS_SESSION_TOKEN"])
        for filename in os.listdir(local_directory):
            if filename.endswith('.pdf'):
                local_path = os.path.join(local_directory, filename)
                s3_key = os.path.join(s3_prefix, filename)
                s3_client.upload_file(local_path, os.environ["BUCKET_NAME"], s3_key)
    except Exception as e:
        print(f"Error uploading PDFs to S3: {e}")

def download_pdf_from_s3(s3_key: str, local_path: str):
    try:
        s3_client = boto3.client('s3',
                                region_name="us-east-1",
                                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                                aws_session_token=os.environ["AWS_SESSION_TOKEN"])
        s3_client.download_file(os.environ["BUCKET_NAME"], s3_key, local_path)
    except Exception as e:
        print(f"Error downloading PDF from S3: {e}")

def list_pdfs_in_local_dir(root_dir='pdf', recursive=True):
    """Return a list of local PDF file paths under `root_dir`.

    Args:
        root_dir (str): Directory to search (default 'pdf').
        recursive (bool): If True, walk subdirectories; otherwise list only top-level files.

    Returns:
        List[str]: Paths to files ending with .pdf (case-insensitive). If directory
        does not exist, returns an empty list.
    """
    if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
        return []

    pdf_paths = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(dirpath, fname))
    else:
        for fname in os.listdir(root_dir):
            if fname.lower().endswith('.pdf') and os.path.isfile(os.path.join(root_dir, fname)):
                pdf_paths.append(os.path.join(root_dir, fname))

    pdf_paths.sort()
    return pdf_paths

def save_to_local_dir(text: str, dir_path: str, file_name: str, overwrite: bool = True) -> str:
    """Save `text` to a file named `file_name` under `dir_path`.

    Args:
        text (str): Text content to write.
        dir_path (str): Target directory to create/use.
        file_name (str): File name to write (can include subpaths).
        overwrite (bool): If False and file exists, raises FileExistsError.

    Returns:
        str: Absolute path to the written file.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Ensure directory exists (handle file_name containing subdirectories)
    full_path = os.path.join(dir_path, file_name)
    parent_dir = os.path.dirname(full_path) or dir_path
    os.makedirs(parent_dir, exist_ok=True)

    if not overwrite and os.path.exists(full_path):
        raise FileExistsError(f"File already exists: {full_path}")

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(text)

    return os.path.abspath(full_path)

if __name__ == "__main__":
    txt = extract_text_from_pdf("pdf" + "/" + "20250124-statements-8506-.pdf")
    print(txt)