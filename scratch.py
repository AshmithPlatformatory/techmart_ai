import json

with open('data/product_catalog_rag.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Sony items:")
for d in data:
    if d['metadata']['brand'] == 'Sony':
        print(d['metadata']['name'], d['metadata']['specs'])

print("\nCamera items:")
for d in data:
    if d['metadata']['category'] == 'Camera':
        print(d['metadata']['name'], d['metadata']['specs'])
