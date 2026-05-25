import urllib.request
from pathlib import Path

out_dir = Path("data/raw/cwru/raw")
out_dir.mkdir(parents=True, exist_ok=True)

base_url = "https://engineering.case.edu/sites/default/files/"

# The exact 16 files needed for early-fault cross-load validation
files = [
    "97.mat", "98.mat", "99.mat", "100.mat",     # Normal (0, 1, 2, 3 HP)
    "105.mat", "106.mat", "107.mat", "108.mat",  # Inner Race 0.007" (0, 1, 2, 3 HP)
    "118.mat", "119.mat", "120.mat", "121.mat",  # Ball 0.007" (0, 1, 2, 3 HP)
    "130.mat", "131.mat", "132.mat", "133.mat"   # Outer Race 0.007" (0, 1, 2, 3 HP)
]

print("Starting CWRU benchmark downloads...")
for f in files:
    out_path = out_dir / f
    if not out_path.exists():
        print(f"Downloading {f}...")
        try:
            # We add a User-Agent header because some academic servers block blank scripts
            req = urllib.request.Request(base_url + f, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"  -> Failed to download {f}: {e}")
    else:
        print(f"{f} already exists. Skipping.")
print("Done!")
