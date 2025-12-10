#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload Models to Cloud Storage
Support for Hugging Face Hub, Google Drive, AWS S3

Usage:
    # Upload to Hugging Face
    python upload_to_cloud.py --provider huggingface --model-path models/phi-2-lora-quantized --repo-name my-phi2-lora
    
    # Upload to Google Drive
    python upload_to_cloud.py --provider gdrive --model-path models/phi-2-lora-quantized
    
    # Upload to AWS S3
    python upload_to_cloud.py --provider s3 --model-path models/phi-2-lora-quantized --bucket-name my-bucket
"""

import os
import json
from pathlib import Path
from typing import Optional
import shutil
import argparse

class CloudUploader:
    """Upload models to cloud storage"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        self.model_size = self.get_directory_size(self.model_path)
    
    def get_directory_size(self, path: Path) -> float:
        """Get directory size in GB"""
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total / (1024**3)
    
    def upload_to_huggingface(self, repo_name: str, private: bool = False):
        """Upload to Hugging Face Hub"""
        print(f"\nUpload to Hugging Face Hub")
        print(f"Model path: {self.model_path}")
        print(f"Repository: {repo_name}")
        print(f"Size: {self.model_size:.2f}GB")
        print(f"Private: {private}")
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            print("\nUploading files...")
            api.upload_folder(
                folder_path=str(self.model_path),
                repo_id=repo_name,
                repo_type="model",
                private=private
            )
            print(f"Successfully uploaded to: https://huggingface.co/{repo_name}")
        except ImportError:
            print("Install huggingface_hub: pip install huggingface_hub")
        except Exception as e:
            print(f"Error: {e}")
    
    def upload_to_gdrive(self, folder_id: Optional[str] = None):
        """Upload to Google Drive"""
        print(f"\nUpload to Google Drive")
        print(f"Model path: {self.model_path}")
        print(f"Size: {self.model_size:.2f}GB")
        
        try:
            from pydrive.auth import GoogleAuth
            from pydrive.drive import GoogleDrive
            
            print("\nAuthenticating with Google Drive...")
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()
            drive = GoogleDrive(gauth)
            
            print("Uploading files...")
            for file_path in self.model_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.model_path)
                    print(f"Uploading: {relative_path}")
                    file_drive = drive.CreateFile(
                        {'title': relative_path.name, 'parents': [{'id': folder_id}]} if folder_id else {'title': relative_path.name}
                    )
                    file_drive.SetContentFile(str(file_path))
                    file_drive.Upload()
            
            print("Successfully uploaded to Google Drive")
        except ImportError:
            print("Install PyDrive: pip install pydrive")
        except Exception as e:
            print(f"Error: {e}")
    
    def upload_to_s3(self, bucket_name: str, prefix: str = ""):
        """Upload to AWS S3"""
        print(f"\nUpload to AWS S3")
        print(f"Model path: {self.model_path}")
        print(f"Bucket: {bucket_name}")
        print(f"Size: {self.model_size:.2f}GB")
        
        try:
            import boto3
            s3 = boto3.client('s3')
            
            print("\nUploading files...")
            for file_path in self.model_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.model_path)
                    s3_key = f"{prefix}/{relative_path}" if prefix else str(relative_path)
                    print(f"Uploading: {s3_key}")
                    s3.upload_file(str(file_path), bucket_name, s3_key)
            
            print(f"Successfully uploaded to s3://{bucket_name}/{prefix}")
        except ImportError:
            print("Install boto3: pip install boto3")
        except Exception as e:
            print(f"Error: {e}")
    
    def create_zip_backup(self, output_path: str):
        """Create local zip backup"""
        print(f"\nCreating local backup...")
        print(f"Output: {output_path}")
        
        shutil.make_archive(output_path.replace('.zip', ''), 'zip', self.model_path.parent, self.model_path.name)
        backup_size = Path(f"{output_path}").stat().st_size / (1024**3)
        print(f"Backup created: {output_path} ({backup_size:.2f}GB)")
    
    def generate_metadata(self):
        """Generate upload metadata"""
        metadata = {
            'model_path': str(self.model_path),
            'size_gb': self.model_size,
            'files': []
        }
        
        for file_path in self.model_path.rglob('*'):
            if file_path.is_file():
                file_size = file_path.stat().st_size / (1024**2)  # MB
                metadata['files'].append({
                    'path': str(file_path.relative_to(self.model_path)),
                    'size_mb': file_size
                })
        
        return metadata

def main():
    parser = argparse.ArgumentParser(description='Upload models to cloud storage')
    parser.add_argument('--provider', choices=['huggingface', 'gdrive', 's3', 'backup'], default='huggingface', help='Cloud provider')
    parser.add_argument('--model-path', required=True, help='Path to model')
    parser.add_argument('--repo-name', help='Repository name (Hugging Face)')
    parser.add_argument('--bucket-name', help='S3 bucket name')
    parser.add_argument('--prefix', default='', help='S3 key prefix')
    parser.add_argument('--folder-id', help='Google Drive folder ID')
    parser.add_argument('--private', action='store_true', help='Make repository private (Hugging Face)')
    
    args = parser.parse_args()
    
    uploader = CloudUploader(args.model_path)
    
    if args.provider == 'huggingface':
        if not args.repo_name:
            raise ValueError("--repo-name is required for Hugging Face upload")
        uploader.upload_to_huggingface(args.repo_name, args.private)
    elif args.provider == 'gdrive':
        uploader.upload_to_gdrive(args.folder_id)
    elif args.provider == 's3':
        if not args.bucket_name:
            raise ValueError("--bucket-name is required for S3 upload")
        uploader.upload_to_s3(args.bucket_name, args.prefix)
    elif args.provider == 'backup':
        output_path = f"{args.model_path}.zip"
        uploader.create_zip_backup(output_path)
    
    metadata = uploader.generate_metadata()
    with open('upload_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: upload_metadata.json")

if __name__ == '__main__':
    main()
