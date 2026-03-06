import os
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil

# CONFIGURATION
# -----------------------------------------------
INPUT_DIR = "data/raw/archive"
OUTPUT_DIR = "data/raw/archive_resized"
TARGET_SIZE = (380, 380) # EfficientNetB4 native resolution
# -----------------------------------------------

def process_image(file_path, output_path):
    try:
        with Image.open(file_path) as img:
            # Convert to RGB (in case of RGBA or Grayscale)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize image
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Save optimized JPEG
            img.save(output_path, 'JPEG', quality=85, optimize=True)
            return True
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return False

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Could not find '{INPUT_DIR}'")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all subfolders (classes)
    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    tasks = []
    
    for cls in classes:
        in_class_dir = Path(INPUT_DIR) / cls
        out_class_dir = Path(OUTPUT_DIR) / cls
        os.makedirs(out_class_dir, exist_ok=True)
        
        # Get all images
        for ext in ('*.jpeg', '*.jpg', '*.png', '*.webp'):
            for img_path in in_class_dir.rglob(ext):
                out_path = out_class_dir / f"{img_path.stem}.jpg"
                tasks.append((img_path, out_path))
                
    total = len(tasks)
    print(f"📦 Found {total} images to compress. Resizing to 380x380...")

    # Process using threads for speed
    success = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        results = executor.map(lambda p: process_image(p[0], p[1]), tasks)
        for idx, result in enumerate(results, 1):
            if result:
                success += 1
            if idx % 500 == 0 or idx == total:
                print(f"⏳ Processed: {idx}/{total} ({idx/total*100:.1f}%)")

    print(f"\n✅ Done! Successfully compressed {success}/{total} images.")
    print(f"📁 Your much smaller dataset is ready at: {OUTPUT_DIR}")
    print("🚀 You can now upload this smaller folder to Google Drive!")

if __name__ == "__main__":
    main()
