import json

def scan_catalog(filepath):
    categories = set()
    brands = set()
    spec_keys = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                meta = item.get("metadata", {})
                if "category" in meta: categories.add(meta["category"])
                if "brand" in meta: brands.add(meta["brand"])
                specs = meta.get("specs", {})
                for k in specs.keys():
                    spec_keys.add(k)
    except Exception as e:
        print(f"Error {filepath}: {e}")
    print(f"--- {filepath} ---")
    print(f"Categories: {list(categories)}")
    print(f"Brands: {list(brands)}")
    print(f"Spec Keys: {list(spec_keys)}\n")

def scan_docs(filepath, is_metadata=False):
    categories = set()
    tags = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if is_metadata:
                    meta = item.get("metadata", {})
                    c = meta.get("category")
                    t = meta.get("tags", [])
                else:
                    c = item.get("category")
                    t = item.get("tags", [])
                
                if c: categories.add(c)
                for tag in t: tags.add(tag)
    except Exception as e:
        print(f"Error {filepath}: {e}")
    print(f"--- {filepath} ---")
    print(f"Categories: {list(categories)}")
    print(f"Tags: {list(tags)}\n")

scan_catalog("data/product_catalog_rag.json")
scan_docs("data/company_faqs.json")
scan_docs("data/company_tos.json")
