"""Deep inspection of archive.zip to find tomato class folders."""
import zipfile
from pathlib import Path

ZIP_PATH = Path(r"C:\Users\JEEVA PRAKASINI G\Downloads\archive.zip")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    members = z.namelist()
    print(f"Total entries: {len(members)}")

    # Collect all unique folder paths at every depth
    all_dirs = sorted(set(
        "/".join(m.split("/")[:d])
        for m in members
        for d in range(1, len(m.split("/")))
        if m.strip()
    ))

    # Show folders that contain "Tomato" or "tomato"
    tomato_dirs = [d for d in all_dirs if "tomato" in d.lower()]
    print(f"\nAll tomato-related folders ({len(tomato_dirs)}):")
    for d in sorted(tomato_dirs):
        prefix = d + "/"
        n_images = sum(
            1 for m in members
            if m.startswith(prefix)
            and not m.endswith("/")
            and m.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"  [{n_images:5d} imgs]  {d}")

    # Also show ALL third-level folders so we can see the full structure
    third_level = sorted(set(
        "/".join(m.split("/")[:3])
        for m in members
        if len(m.split("/")) > 3
    ))
    print(f"\nAll third-level folders ({len(third_level)}) — first 50:")
    for d in third_level[:50]:
        prefix = d + "/"
        n = sum(1 for m in members if m.startswith(prefix) and not m.endswith("/"))
        print(f"  {d}  ({n} files)")
