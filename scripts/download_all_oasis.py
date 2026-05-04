import os
import urllib.request
import tarfile
import subprocess
from pathlib import Path

def download_and_extract(disc_num, base_dir):
    url = f"https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc{disc_num}.tar.gz"
    tar_path = base_dir / f"oasis_cs_freesurfer_disc{disc_num}.tar.gz"
    
    print(f"\n--- Processing Disc {disc_num} ---")
    
    # 1. Download
    if not tar_path.exists():
        print(f"Downloading {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(tar_path, 'wb') as out_file:
                # Read in chunks to show progress
                meta = response.info()
                file_size = int(meta.get("Content-Length", 0))
                
                downloaded = 0
                block_size = 8192
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if file_size > 0:
                        percent = downloaded * 100. / file_size
                        print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {file_size / (1024*1024):.1f} MB)", end="")
            print("\nDownload complete.")
        except Exception as e:
            print(f"\nFailed to download disc {disc_num}: {e}")
            if tar_path.exists():
                tar_path.unlink()
            return False
    else:
        print(f"Archive {tar_path} already exists, skipping download.")
        
    # 2. Extract
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=base_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract disc {disc_num}: {e}")
        return False
        
    # 3. Clean up the tar.gz to save disk space
    print(f"Removing archive {tar_path} to save space...")
    tar_path.unlink()
    
    return True

def main():
    base_dir = Path("data/OASIS-1/FreeSurfer")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Typically there are 11 discs for OASIS-1 FreeSurfer
    # Since disc 1 is already downloaded, we can start from 2
    for i in range(3, 12):
        success = download_and_extract(i, base_dir)
        if not success:
            print(f"Stopping at disc {i} due to an error. (It might be the last disc).")
            # If it's a 404, we just break assuming no more discs
            break
            
    print("\n" + "="*50)
    print("All downloads and extractions finished!")
    print("="*50)
    
    print("\nRunning the preprocessing pipeline on all discs at once...")
    print("This will glob through all subdirectories in data/OASIS-1/FreeSurfer to find brain.mgz files.")
    
    # Run the existing preprocessing script but point it to the root FreeSurfer dir
    # so it captures all discs (disc1, disc2, ..., disc11)
    try:
        subprocess.run([
            "python", "src/data/preprocess_oasis.py",
            "--raw_dir", "data/OASIS-1/FreeSurfer",
            "--processed_dir", "data/processed"
        ], check=True)
        print("\nPreprocessing complete! Your dataset is now enriched with the remaining patients.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during preprocessing: {e}")

if __name__ == "__main__":
    main()
