from pathlib import Path
raw = Path(r"C:\Users\JEEVA PRAKASINI G\Documents\Resume project\LeafIQ\dataset\raw")
for cls in ["Tomato_Healthy","Tomato_Early_Blight","Tomato_Late_Blight","Tomato_Leaf_Mold"]:
    p = raw / cls
    if p.exists():
        files = list(p.iterdir())
        first = files[0].name if files else "empty"
        print(f"{cls}: {len(files)} files  first={first}")
    else:
        print(f"{cls}: MISSING")
