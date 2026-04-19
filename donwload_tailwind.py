import os
import urllib.request

# Create static folder
os.makedirs('static', exist_ok=True)

print("📥 Downloading Tailwind CSS...")

try:
    # Download Tailwind CDN script
    url = "https://cdn.tailwindcss.com"
    output_file = "static/tailwind.js"
    
    urllib.request.urlretrieve(url, output_file)
    
    print(f"✅ Downloaded: {output_file}")
    print(f"📦 File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print("\n✨ Next steps:")
    print("1. Update index.html:")
    print('   Replace: <script src="https://cdn.tailwindcss.com"></script>')
    print('   With:    <script src="{{ url_for(\'static\', filename=\'tailwind.js\') }}"></script>')
    print("\n2. Restart Flask: python app.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Alternative: Use this in index.html instead:")
    print('<link href="https://cdn.jsdelivr.net/npm/tailwindcss@3.4.1/dist/tailwind.min.css" rel="stylesheet">')