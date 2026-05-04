import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

falling_dir = "pose_data/falling"
csv_files = glob.glob(os.path.join(falling_dir, "*.csv"))

if not csv_files:
    print("No falling csv found")
    exit(1)

base_csv = csv_files[0]
print(f"Augmenting {base_csv}...")

df = pd.read_csv(base_csv, header=None)

# 0 is timestamp (string like 2025-11-11T07:06:44.042812)
# 1 is frame idx
# 2 to 133 are x,y,z,v

for i in range(4):
    df_aug = df.copy()
    
    # Generate jitter (mean 0, std 0.005)
    # We only augment the coordinate columns, keep v (visibility) unchanged (v is every 4th col starting from 5: 2=x,3=y,4=z,5=v)
    for col in range(2, 134):
        if col in df_aug.columns:
            # force to numeric
            df_aug[col] = pd.to_numeric(df_aug[col], errors='coerce')
            # Check if it's not a visibility column (5, 9, 13...)
            if (col - 2) % 4 != 3:
                # add noise
                noise = np.random.normal(0, 0.015, size=len(df_aug))
                df_aug[col] += noise
            
    # New timestamp
    now_str = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    
    # Modify timestamps so they look distinct
    df_aug[0] = df_aug[0].astype(str).apply(lambda x: x[:-13] + f"{np.random.randint(10, 99)}.{np.random.randint(100000, 999999)}")
    
    out_name = os.path.join(falling_dir, f"falling_aug_{now_str}_{i}.csv")
    df_aug.to_csv(out_name, header=False, index=False)
    print(f"Created {out_name}")

print("Done generating augmented falling data.")
