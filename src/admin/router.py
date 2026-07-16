import os
import datetime
from fastapi import APIRouter, HTTPException, Security
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.security import APIKeyHeader
from src.db.clickhouse_client import get_client
from src.model_loader import get_sentence_transformer

admin_router = APIRouter(prefix="/admin")
html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")

api_key_header = APIKeyHeader(name="X-API-Key")

def get_admin_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.environ.get("ADMIN_API_KEY", "dev-secret-key")
    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

class QueryData(BaseModel):
    table: str

class SaveData(BaseModel):
    table: str
    rows: List[Dict[str, Any]]

@admin_router.get("/", response_class=HTMLResponse)
async def get_admin_page():
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@admin_router.post("/api/data")
async def get_data(data: QueryData, api_key: str = Security(get_admin_api_key)):
    client = get_client()
    if data.table not in ["product_catalog", "company_faqs", "company_tos"]:
        raise HTTPException(status_code=400, detail="Invalid table")
    
    result = client.query(f"SELECT * FROM {data.table}")
    rows = []
    if result.result_rows:
        for row in result.result_rows:
            row_dict = dict(zip(result.column_names, row))
            if "embedding" in row_dict:
                row_dict.pop("embedding")
            if "question_embedding" in row_dict:
                row_dict.pop("question_embedding")
            rows.append(row_dict)
    
    return {"columns": list(rows[0].keys()) if rows else [], "data": rows}

@admin_router.post("/api/save")
async def save_data(data: SaveData, api_key: str = Security(get_admin_api_key)):
    client = get_client()
    table = data.table
    rows = data.rows

    id_col = "product_id" if table == "product_catalog" else ("faq_id" if table == "company_faqs" else "doc_id")
    
    id_query = f"SELECT {id_col} FROM {table}"
    id_result = client.query(id_query)
    current_max = 0
    if id_result.result_rows:
        for row in id_result.result_rows:
            try:
                parts = str(row[0]).split("-")
                num = int(parts[-1])
                if num > current_max:
                    current_max = num
            except:
                pass

    for r in rows:
        val = str(r.get(id_col, "")).strip()
        if not val or val == "None" or val == "":
            current_max += 1
            if table == "product_catalog":
                cat = str(r.get("category", "")).strip()
                cat_map = {
                    "Laptop": "TK-LAP",
                    "CPU": "TK-CPU",
                    "Television": "TK-TV",
                    "Phone": "TK-PHN",
                    "Monitor": "TK-MON",
                    "GPU": "TK-GPU",
                    "Audio": "TK-AUD",
                    "Accessories": "TK-ACC",
                    "Tablet": "TK-TAB",
                    "Console": "TK-CON",
                    "Camera": "TK-CAM",
                    "Smartwatch": "TK-WTC",
                    "Printer": "TK-PRN",
                    "Router": "TK-RTR"
                }
                prefix = cat_map.get(cat, f"TK-{cat[:3].upper()}" if cat else "TK-PRD")
                r[id_col] = f"{prefix}-{current_max:03d}"
            elif table == "company_faqs":
                r[id_col] = f"FAQ-{current_max:03d}"
            elif table == "company_tos":
                r[id_col] = f"TOS-{current_max:03d}"

        if table == "product_catalog":
            r["price_inr"] = float(r.get("price_inr") or 0.0)
            r["stock_qty"] = int(r.get("stock_qty") or 0)
            r["warranty_months"] = int(r.get("warranty_months") or 0)
            r["rating"] = float(r.get("rating") or 0.0)
            r["in_stock"] = int(r.get("in_stock") or 0)
            if "name" in r:
                r["embedding"] = get_sentence_transformer().encode(r["name"]).tolist()
        elif table == "company_faqs":
            tags_val = r.get("tags", "")
            if isinstance(tags_val, str):
                r["tags"] = [t.strip() for t in tags_val.split(",") if t.strip()]
            if "question" in r:
                r["question_embedding"] = get_sentence_transformer().encode(r["question"]).tolist()
        elif table == "company_tos":
            tags_val = r.get("tags", "")
            if isinstance(tags_val, str):
                r["tags"] = [t.strip() for t in tags_val.split(",") if t.strip()]
            if "effective_date" in r and isinstance(r["effective_date"], str):
                try:
                    r["effective_date"] = datetime.datetime.strptime(r["effective_date"], "%Y-%m-%d").date()
                except ValueError:
                    pass
            if "last_updated" in r and isinstance(r["last_updated"], str):
                try:
                    r["last_updated"] = datetime.datetime.strptime(r["last_updated"], "%Y-%m-%d").date()
                except ValueError:
                    pass

    if not rows:
        client.command(f"TRUNCATE TABLE {table}")
        return {"status": "success"}

    columns = list(rows[0].keys())
    insert_data = [[r.get(c) for c in columns] for r in rows]

    # Shadow table swap
    temp_table = f"{table}_temp"
    client.command(f"CREATE TABLE {temp_table} AS {table}")
    client.insert(temp_table, insert_data, column_names=columns)
    
    client.command(f"EXCHANGE TABLES {table} AND {temp_table}")
    client.command(f"DROP TABLE {temp_table}")
    
    return {"status": "success"}