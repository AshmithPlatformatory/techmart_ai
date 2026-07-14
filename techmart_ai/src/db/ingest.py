import json
import os
import datetime
from sentence_transformers import SentenceTransformer
from src.db.clickhouse_client import get_client
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


model = SentenceTransformer('all-MiniLM-L6-v2')

def load_json(filename):
    filepath = os.path.join("data", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def ingest_data():
    client = get_client()

    logger.info("Ingesting Customers...")
    customers = load_json("customers.json")
    for c in customers:
        c["is_active"] = 1 if c.get("is_active") else 0
  
    customer_cols = list(customers[0].keys())
    customer_data = [[c.get(col) for col in customer_cols] for c in customers]
    client.insert("customers", customer_data, column_names=customer_cols)

    logger.info("Ingesting Order History...")
    orders = load_json("order_history.json")
    for o in orders:
        o["items"] = json.dumps(o.get("items", []))
    order_cols = list(orders[0].keys())
    order_data = [[o.get(col) for col in order_cols] for o in orders]
    client.insert("order_history", order_data, column_names=order_cols)

    logger.info("Ingesting Company FAQs...")
    faqs = load_json("company_faqs.json")
    for f in faqs:
        f["question_embedding"] = model.encode(f["question"]).tolist()
    faq_cols = list(faqs[0].keys())
    faq_data = [[f.get(col) for col in faq_cols] for f in faqs]
    client.insert("company_faqs", faq_data, column_names=faq_cols)

    logger.info("Ingesting Company TOS...")
    tos = load_json("company_tos.json")
    for t in tos:
        t["effective_date"] = datetime.datetime.strptime(t["effective_date"], "%Y-%m-%d").date()
        t["last_updated"] = datetime.datetime.strptime(t["last_updated"], "%Y-%m-%d").date()
    tos_cols = list(tos[0].keys())
    tos_data = [[t.get(col) for col in tos_cols] for t in tos]
    client.insert("company_tos", tos_data, column_names=tos_cols)

    logger.info("Ingesting Product Catalog...")
    raw_catalog = load_json("product_catalog_rag.json")
    catalog_data = []
    for p in raw_catalog:
        meta = p.get("metadata", {})
        row = {
            "product_id": p.get("product_id"),
            "page_content": p.get("page_content"),
            "name": meta.get("name"),
            "brand": meta.get("brand"),
            "category": meta.get("category"),
            "price_inr": float(meta.get("price_inr", 0)),
            "stock_qty": int(meta.get("stock_qty", 0)),
            "warranty_months": int(meta.get("warranty_months", 0)),
            "rating": float(meta.get("rating", 0)),
            "in_stock": 1 if meta.get("in_stock") else 0,
            "price_tier": meta.get("price_tier"),
            "embedding": model.encode(meta.get("name", p.get("page_content"))).tolist()
        }
        catalog_data.append(row)
    
    if catalog_data:
        cat_cols = list(catalog_data[0].keys())
        cat_data = [[r.get(col) for col in cat_cols] for r in catalog_data]
        client.insert("product_catalog", cat_data, column_names=cat_cols)

    logger.info("Initial data ingestion complete.")

if __name__ == "__main__":
    ingest_data()