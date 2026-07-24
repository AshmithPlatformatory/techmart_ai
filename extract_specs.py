import json

def get_spec_keys(filepath):
    keys = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                specs = item.get("metadata", {}).get("specs", {})
                for k in specs.keys():
                    keys.add(k)
    except Exception as e:
        print(f"Error {filepath}: {e}")
    return list(keys)

print("SPEC_KEYS:", get_spec_keys("data/product_catalog_rag.json"))
