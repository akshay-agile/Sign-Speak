# import os
# import json
# import shutil

# # Paths to your dataset
# DATASET_PATH = "ISL_CSLRT_Corpus"
# WORD_IMAGES_PATH = os.path.join(DATASET_PATH, "Frames_Word_Level")
# SENTENCE_VIDEOS_PATH = os.path.join(DATASET_PATH, "Videos_Sentence_Level")
# OUTPUT_IMAGE_DIR = "static/sign_images"
# OUTPUT_VIDEO_DIR = "static/sign_videos"
# INDEX_FILE = "sign_index.json"

# def normalize_name(name):
#     """Normalize sign names for consistency"""
#     return name.lower().replace(' ', '_').replace('-', '_')

# def index_dataset():
#     """Index all available signs from both images and videos"""
    
#     if not os.path.exists(DATASET_PATH):
#         print(f"❌ Dataset not found at: {DATASET_PATH}")
#         print("Please update DATASET_PATH in the script")
#         return
    
#     # Create output directories
#     os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
#     os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    
#     sign_index = {}
    
#     print("📥 Processing Word-Level Images...")
#     # Process word-level images
#     if os.path.exists(WORD_IMAGES_PATH):
#         word_folders = [f for f in os.listdir(WORD_IMAGES_PATH)
#                        if os.path.isdir(os.path.join(WORD_IMAGES_PATH, f))]
        
#         for word_folder in sorted(word_folders):
#             word_path = os.path.join(WORD_IMAGES_PATH, word_folder)
            
#             # Get all images in this word folder
#             images = [f for f in os.listdir(word_path)
#                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
#             if images:
#                 # Use first image as representative
#                 source_image = os.path.join(word_path, images[0])
#                 normalized_name = normalize_name(word_folder)
#                 dest_image = os.path.join(OUTPUT_IMAGE_DIR, f"{normalized_name}.jpg")
                
#                 # Copy image
#                 shutil.copy2(source_image, dest_image)
                
#                 # Add to index
#                 sign_index[normalized_name] = {
#                     'original_name': word_folder,
#                     'type': 'image',
#                     'image_path': f"/static/sign_images/{normalized_name}.jpg",
#                     'image_count': len(images)
#                 }
        
#         print(f"✅ Processed {len(sign_index)} word images")
    
#     print("\n📥 Processing Sentence-Level Videos...")
#     # Process sentence-level videos (extract individual words where possible)
#     if os.path.exists(SENTENCE_VIDEOS_PATH):
#         sentence_folders = [f for f in os.listdir(SENTENCE_VIDEOS_PATH)
#                           if os.path.isdir(os.path.join(SENTENCE_VIDEOS_PATH, f))]
        
#         video_count = 0
#         for sentence_folder in sorted(sentence_folders):
#             sentence_path = os.path.join(SENTENCE_VIDEOS_PATH, sentence_folder)
            
#             # Get all videos in this folder
#             videos = [f for f in os.listdir(sentence_path)
#                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            
#             if videos:
#                 # Try to extract individual word videos
#                 # Example: "comb your hair" folder might have "comb (2).mp4", "comb your hair (2).mp4"
                
#                 for video_file in videos:
#                     video_name = os.path.splitext(video_file)[0]
#                     # Remove numbers in parentheses
#                     clean_name = video_name.split('(')[0].strip()
#                     normalized_name = normalize_name(clean_name)
                    
#                     source_video = os.path.join(sentence_path, video_file)
#                     dest_video = os.path.join(OUTPUT_VIDEO_DIR, f"{normalized_name}.mp4")
                    
#                     # Only copy if we don't already have this video or it's a better match
#                     if normalized_name not in sign_index or sign_index[normalized_name]['type'] == 'image':
#                         # Copy video
#                         shutil.copy2(source_video, dest_video)
                        
#                         # Update index
#                         if normalized_name in sign_index:
#                             # Upgrade from image to video
#                             sign_index[normalized_name]['type'] = 'video'
#                             sign_index[normalized_name]['video_path'] = f"/static/sign_videos/{normalized_name}.mp4"
#                         else:
#                             sign_index[normalized_name] = {
#                                 'original_name': clean_name,
#                                 'type': 'video',
#                                 'video_path': f"/static/sign_videos/{normalized_name}.mp4"
#                             }
                        
#                         video_count += 1
        
#         print(f"✅ Processed {video_count} videos")
    
#     # Save index
#     with open(INDEX_FILE, 'w') as f:
#         json.dump(sign_index, f, indent=2)
    
#     # Statistics
#     image_only = sum(1 for s in sign_index.values() if s['type'] == 'image')
#     video_signs = sum(1 for s in sign_index.values() if s['type'] == 'video')
    
#     print(f"\n✅ Indexing complete!")
#     print(f"📊 Total unique signs: {len(sign_index)}")
#     print(f"📷 Image signs: {image_only}")
#     print(f"🎬 Video signs: {video_signs}")
#     print(f"📁 Images saved to: {OUTPUT_IMAGE_DIR}")
#     print(f"📁 Videos saved to: {OUTPUT_VIDEO_DIR}")
#     print(f"📄 Index saved to: {INDEX_FILE}")
    
#     # Show sample signs
#     sample_signs = list(sign_index.keys())[:15]
#     print(f"\n🔤 Sample signs: {', '.join(sample_signs)}")

# if __name__ == "__main__":
#     index_dataset()






import os
import json
import shutil
import re

# Paths to your dataset
DATASET_PATH = "ISL_CSLRT_Corpus"
WORD_IMAGES_PATH = os.path.join(DATASET_PATH, "Frames_Word_Level")
SENTENCE_VIDEOS_PATH = os.path.join(DATASET_PATH, "Videos_Sentence_Level")
OUTPUT_IMAGE_DIR = "static/sign_images"
OUTPUT_VIDEO_DIR = "static/sign_videos"
INDEX_FILE = "sign_index.json"


def normalize_name(name):
    """Normalize sign names for consistency in keys"""
    return name.lower().replace(' ', '_').replace('-', '_')


def display_name(name):
    """Convert to user-friendly display name by replacing _ and - with space"""
    name_with_spaces = re.sub(r'[-_]+', ' ', name)
    return ' '.join(name_with_spaces.split())


def index_dataset():
    """Index all available signs from both images and videos"""

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("Please update DATASET_PATH in the script")
        return

    # Create output directories
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    sign_index = {}

    print("📥 Processing Word-Level Images...")
    # Process word-level images
    if os.path.exists(WORD_IMAGES_PATH):
        word_folders = [f for f in os.listdir(WORD_IMAGES_PATH)
                        if os.path.isdir(os.path.join(WORD_IMAGES_PATH, f))]

        for word_folder in sorted(word_folders):
            word_path = os.path.join(WORD_IMAGES_PATH, word_folder)

            # Get all images in this word folder
            images = [f for f in os.listdir(word_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if images:
                # Use first image as representative
                source_image = os.path.join(word_path, images[0])
                normalized_name = normalize_name(word_folder)
                dest_image = os.path.join(OUTPUT_IMAGE_DIR, f"{normalized_name}.jpg")

                # Copy image
                shutil.copy2(source_image, dest_image)

                # Add to index with display name replacing _ and - with spaces
                sign_index[normalized_name] = {
                    'original_name': display_name(word_folder),
                    'type': 'image',
                    'image_path': f"/static/sign_images/{normalized_name}.jpg",
                    'image_count': len(images)
                }

        print(f"✅ Processed {len(sign_index)} word images")

    print("\n📥 Processing Sentence-Level Videos...")
    # Process sentence-level videos (extract individual words where possible)
    if os.path.exists(SENTENCE_VIDEOS_PATH):
        sentence_folders = [f for f in os.listdir(SENTENCE_VIDEOS_PATH)
                            if os.path.isdir(os.path.join(SENTENCE_VIDEOS_PATH, f))]

        video_count = 0
        for sentence_folder in sorted(sentence_folders):
            sentence_path = os.path.join(SENTENCE_VIDEOS_PATH, sentence_folder)

            # Get all videos in this folder
            videos = [f for f in os.listdir(sentence_path)
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]

            if videos:
                # Try to extract individual word videos
                # Example: "comb your hair" folder might have "comb (2).mp4", "comb your hair (2).mp4"

                for video_file in videos:
                    video_name = os.path.splitext(video_file)[0]
                    # Remove numbers in parentheses
                    clean_name = video_name.split('(')[0].strip()
                    normalized_name = normalize_name(clean_name)

                    source_video = os.path.join(sentence_path, video_file)
                    dest_video = os.path.join(OUTPUT_VIDEO_DIR, f"{normalized_name}.mp4")

                    # Only copy if we don't already have this video or it's a better match
                    if normalized_name not in sign_index or sign_index[normalized_name]['type'] == 'image':
                        # Copy video
                        shutil.copy2(source_video, dest_video)

                        # Update index with display name replacing _ and - with spaces
                        if normalized_name in sign_index:
                            # Upgrade from image to video
                            sign_index[normalized_name]['type'] = 'video'
                            sign_index[normalized_name]['video_path'] = f"/static/sign_videos/{normalized_name}.mp4"
                            sign_index[normalized_name]['original_name'] = display_name(clean_name)
                        else:
                            sign_index[normalized_name] = {
                                'original_name': display_name(clean_name),
                                'type': 'video',
                                'video_path': f"/static/sign_videos/{normalized_name}.mp4"
                            }

                        video_count += 1

        print(f"✅ Processed {video_count} videos")

    # Save index
    with open(INDEX_FILE, 'w') as f:
        json.dump(sign_index, f, indent=2)

    # Statistics
    image_only = sum(1 for s in sign_index.values() if s['type'] == 'image')
    video_signs = sum(1 for s in sign_index.values() if s['type'] == 'video')

    print(f"\n✅ Indexing complete!")
    print(f"📊 Total unique signs: {len(sign_index)}")
    print(f"📷 Image signs: {image_only}")
    print(f"🎬 Video signs: {video_signs}")
    print(f"📁 Images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"📁 Videos saved to: {OUTPUT_VIDEO_DIR}")
    print(f"📄 Index saved to: {INDEX_FILE}")

    # Show sample signs
    sample_signs = list(sign_index.keys())[:15]
    print(f"\n🔤 Sample signs: {', '.join(sample_signs)}")


if __name__ == "__main__":
    index_dataset()
