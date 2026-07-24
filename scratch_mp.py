import json

with open('data/product_catalog_rag.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

out = []
for d in data:
    if d['metadata']['category'] == 'Camera':
        out.append(f"{d['metadata']['name']} - {d['metadata']['specs'].get('camera_mp')}")

with open('scratch_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
