# Use the "files" URL (not /preview/) and add ?download=1
URL="https://zenodo.org/record/15857116/files/TEXTAR-CHECKPOINTS.zip?download=1"
DEST="/code/"

# 1) Download the real ZIP
wget -O weights.zip "$URL"

# 2) Make sure it's a valid zip
file weights.zip
# You should see something like: Zip archive data, at least vX.Y to extract

# 3) Unpack it
mkdir -p "$DEST"
unzip -q weights.zip -d "$DEST"

# 4) Clean up
rm weights.zip