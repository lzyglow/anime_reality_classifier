import argparse
import os
import sys
import time
import tarfile
import pandas as pd
from pathlib import Path


def extract_tar(tar_path, extract_dir):
    """Extract tar file to directory."""
    print(f"📦 Extracting {tar_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_dir)
    
    # Count extracted files
    file_count = sum(1 for _ in Path(extract_dir).rglob('*') if _.is_file())
    print(f"   Extracted {file_count} files")
    return extract_dir

def main():
    parser = argparse.ArgumentParser(
        description="Anime vs Realistic Image Classification - Challenge 1"
    )
    
    parser.add_argument(
        "tar_files",
        nargs="+",
        help="Paths to .tar files (e.g., zip1.tar)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--model-path",
        default="best_model.pth",
        help="Path to trained model (default: best_model.pth)"
    )
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    print("="*60)
    print("CHALLENGE 1: ANIME vs REALISTIC CLASSIFICATION")
    print("="*60)
    
    # Check tar files exist
    for tar_file in args.tar_files:
        if not os.path.exists(tar_file):
            print(f"❌ Error: Tar file not found: {tar_file}")
            sys.exit(1)
    
    print(f"📁 Processing {len(args.tar_files)} tar file(s):")
    for tf in args.tar_files:
        print(f"   • {tf}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Import your inference code
    # Make sure we can import from src
    src_dir = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_dir))
    
    try:
        from src.pipeline import run_inference
        print("✅ Successfully imported pipeline module")
    except ImportError as e:
        print(f"❌ Error importing pipeline: {e}")
        print(f"   Current Python path: {sys.path}")
        print(f"   Looking for module in: {src_dir}")
        sys.exit(1)
    
    # Process each tar file
    total_images = 0
    total_anime = 0
    total_realistic = 0
    all_predictions = []
    
    for tar_idx, tar_file in enumerate(args.tar_files):
        print(f"\n{'='*40}")
        print(f"Processing: {os.path.basename(tar_file)}")
        print(f"{'='*40}")
        
        # Extract tar file
        tar_name = Path(tar_file).stem
        extract_dir = os.path.join(args.output_dir, "extracted", tar_name)
        extracted_path = extract_tar(tar_file, extract_dir)
        
        # Run inference
        print(f"\n🔍 Running inference on extracted images...")
        
        # Modify your pipeline to return the DataFrame
        pred_df = run_inference(
            image_dir=extracted_path,
            model_path=args.model_path,
        )
        
        # Save predictions for this tar file
        output_file = os.path.join(args.output_dir, f"{tar_name}_predictions.parquet")
        pred_df.to_parquet(output_file, index=False)
        print(f"💾 Saved: {output_file}")
        
        # Calculate statistics
        total = len(pred_df)
        anime_count = (pred_df['label'] == 'anime').sum()
        realistic_count = (pred_df['label'] == 'realistic').sum()
        error_count = (pred_df['label'] == 'error').sum() if 'error' in pred_df['label'].values else 0
        
        # Update totals
        total_images += total
        total_anime += anime_count
        total_realistic += realistic_count
        
        # Print per-file statistics
        print(f"\n📊 Results for {tar_name}:")
        print(f"   Total images: {total:,}")
        print(f"   Anime: {anime_count:,} ({anime_count/total*100:.1f}%)")
        print(f"   Realistic: {realistic_count:,} ({realistic_count/total*100:.1f}%)")
        print(f"   Errors: {error_count:,}")
        
        # Add tar file identifier
        pred_df['tar_file'] = tar_name
        all_predictions.append(pred_df)
        
        # Clean up extracted files (optional)
        # import shutil
        # shutil.rmtree(extract_dir, ignore_errors=True)
        # print(f"   Cleaned up temporary files")
    
    # Combine all predictions
    if all_predictions:
        combined_df = pd.concat(all_predictions, ignore_index=True)
        combined_output = os.path.join(args.output_dir, "all_predictions.parquet")
        combined_df.to_parquet(combined_output, index=False)
        print(f"\n💾 Combined predictions saved to: {combined_output}")
    
    # Print overall statistics
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("🎯 PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"⏱️  Total runtime: {total_time:.1f} seconds")
    print(f"📦 Total tar files processed: {len(args.tar_files)}")
    
    if total_images > 0:
        anime_percent = total_anime / total_images * 100
        realistic_percent = total_realistic / total_images * 100
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total images classified: {total_images:,}")
        print(f"   Anime: {total_anime:,} ({anime_percent:.1f}%)")
        print(f"   Realistic: {total_realistic:,} ({realistic_percent:.1f}%)")
        print(f"   Speed: {total_images/total_time:.1f} images/second")
    
    print(f"\n💾 Results saved to: {args.output_dir}")
    
    # Save summary file
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Challenge 1: Anime vs Realistic Classification\n")
        f.write(f"Generated: {time.ctime()}\n")
        f.write(f"Total runtime: {total_time:.1f} seconds\n")
        f.write(f"Total tar files: {len(args.tar_files)}\n")
        f.write(f"Total images: {total_images:,}\n")
        if total_images > 0:
            f.write(f"Anime: {total_anime:,} ({anime_percent:.1f}%)\n")
            f.write(f"Realistic: {total_realistic:,} ({realistic_percent:.1f}%)\n")
    
    print(f"\n✅ Done!")

if __name__ == "__main__":
    main()