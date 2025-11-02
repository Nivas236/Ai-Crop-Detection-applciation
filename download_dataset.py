"""
Helper script to download and organize the Plant Disease Dataset from Kaggle
Uses kagglehub library (simpler than Kaggle CLI)
Reference: https://github.com/Kaggle/kagglehub
"""

import kagglehub
import os
import shutil
from pathlib import Path
import zipfile

def check_kagglehub_installed():
    """Check if kagglehub is installed"""
    try:
        import kagglehub
        print("✅ kagglehub is installed")
        return True
    except ImportError:
        print("❌ kagglehub is not installed")
        print("   Please run: pip install kagglehub")
        return False

def authenticate_kaggle():
    """Authenticate with Kaggle"""
    print("\n🔐 Authenticating with Kaggle...")
    
    # Check if credentials file exists
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Found kaggle.json credentials file")
        print("   Authentication will use this file automatically")
        return True
    else:
        print("⚠️  No kaggle.json found")
        print("   Options:")
        print("   1. Place kaggle.json at:", kaggle_json)
        print("      (Download from: https://www.kaggle.com/settings → 'Create New API Token')")
        print("   2. Use interactive login (will prompt you for username and token)")
        
        choice = input("\n   Use interactive login? (y/n): ").strip().lower()
        if choice == 'y':
            try:
                kagglehub.login()
                print("✅ Authentication successful!")
                return True
            except Exception as e:
                print(f"❌ Authentication failed: {e}")
                return False
        else:
            print("\n💡 To get kaggle.json:")
            print("   1. Go to https://www.kaggle.com")
            print("   2. Sign in → Profile → Account")
            print("   3. Scroll to 'API' section")
            print("   4. Click 'Create New API Token'")
            print("   5. Place downloaded kaggle.json at:", kaggle_json)
            return False

def download_dataset(dataset_name):
    """Download dataset from Kaggle using kagglehub"""
    print(f"\n📥 Downloading dataset: {dataset_name}")
    print("   This may take a few minutes (dataset is usually 1-2 GB)...")
    
    try:
        # Download dataset - kagglehub handles everything!
        path = kagglehub.dataset_download(dataset_name)
        print(f"✅ Download completed!")
        print(f"   Dataset saved to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Make sure you're authenticated (run kagglehub.login() if needed)")
        print("   - Check that the dataset name is correct")
        print("   - Some datasets require accepting competition rules first")
        print("   - Verify your internet connection")
        return None

def find_train_valid_folders(dataset_path):
    """Find train and valid folders in the downloaded dataset"""
    print(f"\n📁 Looking for train/valid folders in: {dataset_path}")
    
    # Common folder names
    possible_train_names = ["train", "Train", "TRAIN", "training", "Training"]
    possible_valid_names = ["valid", "Valid", "VALID", "validation", "Validation", "Validation_Set", "test", "Test"]
    
    train_folder = None
    valid_folder = None
    
    dataset_path = Path(dataset_path)
    
    # Search in the main directory
    for item in dataset_path.iterdir():
        if item.is_dir():
            if item.name in possible_train_names:
                train_folder = item
            elif item.name in possible_valid_names:
                valid_folder = item
    
    # Also check one level deeper (some datasets have an extra wrapper folder)
    for subdir in dataset_path.iterdir():
        if subdir.is_dir():
            for item in subdir.iterdir():
                if item.is_dir():
                    if item.name in possible_train_names and train_folder is None:
                        train_folder = item
                    elif item.name in possible_valid_names and valid_folder is None:
                        valid_folder = item
    
    return train_folder, valid_folder

def organize_dataset(dataset_path):
    """Organize extracted files into Datasets/train and Datasets/valid"""
    print("\n📁 Organizing dataset structure...")
    
    # Create target directories
    datasets_dir = Path("Datasets")
    train_dir = datasets_dir / "train"
    valid_dir = datasets_dir / "valid"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    
    # Find train and valid folders
    train_source, valid_source = find_train_valid_folders(dataset_path)
    
    # Copy train folder
    if train_source and train_source.exists():
        print(f"   ✅ Found training data: {train_source}")
        
        # Count classes
        class_folders = [d for d in train_source.iterdir() if d.is_dir()]
        print(f"   📊 Found {len(class_folders)} class folders")
        
        # Copy all subdirectories (class folders)
        for class_folder in class_folders:
            dest = train_dir / class_folder.name
            if not dest.exists():
                print(f"      Copying: {class_folder.name}...")
                shutil.copytree(class_folder, dest)
            else:
                print(f"      ⚠️  Skipping (already exists): {class_folder.name}")
        
        print(f"✅ Training data organized in: {train_dir}")
    else:
        print(f"⚠️  Could not find training folder automatically")
        print(f"   Please manually copy train folders to: {train_dir}")
    
    # Copy valid folder
    if valid_source and valid_source.exists():
        print(f"   ✅ Found validation data: {valid_source}")
        
        # Count classes
        class_folders = [d for d in valid_source.iterdir() if d.is_dir()]
        print(f"   📊 Found {len(class_folders)} class folders")
        
        # Copy all subdirectories (class folders)
        for class_folder in class_folders:
            dest = valid_dir / class_folder.name
            if not dest.exists():
                print(f"      Copying: {class_folder.name}...")
                shutil.copytree(class_folder, dest)
            else:
                print(f"      ⚠️  Skipping (already exists): {class_folder.name}")
        
        print(f"✅ Validation data organized in: {valid_dir}")
    else:
        print(f"⚠️  Could not find validation folder automatically")
        print(f"   Please manually copy valid folders to: {valid_dir}")
    
    # Check final structure
    train_subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
    valid_subdirs = [d for d in valid_dir.iterdir() if d.is_dir()]
    
    print(f"\n📊 Final structure:")
    print(f"   Train folders: {len(train_subdirs)}")
    print(f"   Valid folders: {len(valid_subdirs)}")
    
    if len(train_subdirs) >= 30:  # Should be ~38
        print("✅ Dataset structure looks good!")
        return True
    else:
        print("⚠️  Expected ~38 class folders. Please verify manually.")
        print("\n💡 If folders are missing, check the original dataset path:")
        print(f"   {dataset_path}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("  Plant Disease Dataset Downloader (using kagglehub)")
    print("=" * 60)
    print("\nReference: https://github.com/Kaggle/kagglehub")
    
    # Check prerequisites
    if not check_kagglehub_installed():
        return
    
    # Authenticate
    if not authenticate_kaggle():
        print("\n⚠️  Authentication is required to download datasets")
        return
    
    # Dataset name - UPDATE THIS with the actual dataset name
    # Common ones:
    # - "vipoooool/new-plant-diseases-dataset"
    # - "abdallahalidev/plantvillage-dataset"
    # - "d/arjuntejaswi/plant-disease"
    
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print("To find the correct dataset name:")
    print("1. Go to https://www.kaggle.com/datasets")
    print("2. Search for 'New Plant Diseases Dataset'")
    print("3. Click on the dataset")
    print("4. Look at the URL: kaggle.com/datasets/<username>/<dataset-name>")
    print("5. Use format: <username>/<dataset-name>")
    print("=" * 60)
    
    # Default dataset name (common one)
    default_dataset = "vipoooool/new-plant-diseases-dataset"
    
    print(f"\n📌 Using default dataset: {default_dataset}")
    use_default = input("   Use this? (y/n): ").strip().lower()
    
    if use_default != 'y':
        dataset_name = input("\n   Enter dataset name (format: username/dataset-name): ").strip()
        if not dataset_name:
            print("❌ No dataset name provided. Exiting.")
            return
    else:
        dataset_name = default_dataset
    
    print(f"\n📥 Preparing to download: {dataset_name}")
    confirm = input("   Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Cancelled by user.")
        return
    
    # Download
    dataset_path = download_dataset(dataset_name)
    
    if not dataset_path:
        return
    
    # Organize
    success = organize_dataset(dataset_path)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Dataset download and organization complete!")
    else:
        print("⚠️  Download complete, but please verify the structure manually")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify Datasets/train/ has ~38 class subfolders")
    print("2. Verify Datasets/valid/ has ~38 class subfolders")
    print("3. Run crop-detection.ipynb to train the model")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
