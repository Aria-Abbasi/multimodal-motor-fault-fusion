import pandas as pd
meta = pd.read_csv('data/metadata/metadata_master.csv')
split = pd.read_csv('data/splits/paderborn_condition_generalization.csv')

meta_ids = set(meta[meta['dataset']=='paderborn']['recording_id'])
split_ids = set(split['recording_id'])
overlap = meta_ids.intersection(split_ids)

print(f"IDs in Metadata: {len(meta_ids)}")
print(f"IDs in Split File: {len(split_ids)}")
print(f"Overlap (Matches): {len(overlap)}")

if len(overlap) > 0:
    print(f"Sample Matching ID: {list(overlap)[0]}")
else:
    print("\nCRITICAL: Zero matches found.")
    print(f"Metadata Sample: {list(meta_ids)[0] if meta_ids else 'None'}")
    print(f"Split File Sample: {list(split_ids)[0] if split_ids else 'None'}")
