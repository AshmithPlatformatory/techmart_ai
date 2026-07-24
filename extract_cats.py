import json

def get_unique_categories(filepath, is_metadata=False):
    cats = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if is_metadata:
                    c = item.get("metadata", {}).get("category")
                else:
                    c = item.get("category")
                if c:
                    cats.add(c)
    except Exception as e:
        print(f"Error {filepath}: {e}")
    return list(cats)

print("TOS:", get_unique_categories("data/company_tos.json"))
print("FAQ:", get_unique_categories("data/company_faqs.json"))
print("CATALOG:", get_unique_categories("data/product_catalog_rag.json", is_metadata=True))
