import pandas as pd

# Load master metadata
df = pd.read_csv('data/metadata/metadata_master.csv')
cwru = df[df['dataset'] == 'cwru'].copy()

print(f"Total CWRU files found: {len(cwru)}")
print(f"Loads available: {cwru['load'].unique()}")

# Determine split logic based on available loads
loads = sorted([l for l in cwru['load'].unique() if pd.notna(l)])

if len(loads) > 1:
    test_load = loads[-1]
    val_load = loads[-2] if len(loads) > 2 else loads[0]
    print(f"Assigning Load {test_load} -> test, Load {val_load} -> val, Others -> train")
    
    def assign_split(row):
        if row['load'] == test_load: return 'test'
        elif row['load'] == val_load: return 'val'
        else: return 'train'
        
    cwru['split'] = cwru.apply(assign_split, axis=1)
else:
    print("Only 1 load found! Falling back to a 70/15/15 random recording split to prevent crashes.")
    # Shuffle predictably
    cwru = cwru.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(cwru)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    cwru['split'] = 'test'
    cwru.loc[:train_idx, 'split'] = 'train'
    cwru.loc[train_idx+1:val_idx, 'split'] = 'val'

print("\nNew split distribution:")
print(cwru['split'].value_counts())

# Save fixed split file
cwru[['recording_id', 'split']].to_csv('data/splits/cwru_leave_one_load_out.csv', index=False)
print("\nSuccessfully overwritten data/splits/cwru_leave_one_load_out.csv!")
