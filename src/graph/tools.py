from sentence_transformers import SentenceTransformer
from src.db.clickhouse_client import get_client

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_support_context(query: str) -> str:
    client = get_client()
    vector = model.encode(query).tolist()
    res = client.query(f"""
        SELECT question, answer, related_tos, cosineDistance(question_embedding, {vector}) as dist
        FROM company_faqs
        ORDER BY dist ASC LIMIT 2
    """)
    out = []
    if res.result_rows:
        for r in res.result_rows:
            out.append(f"FAQ: {r[0]} | Answer: {r[1]} | TOS Ref: {r[2]}")
            if r[2]:
                tos_res = client.query(f"SELECT title, content FROM company_tos WHERE doc_id = '{r[2]}'")
                if tos_res.result_rows:
                    out.append(f"TOS ({r[2]}): {tos_res.result_rows[0][0]} - {tos_res.result_rows[0][1]}")
    return "\n".join(out)

def get_catalog_context(query: str) -> str:
    client = get_client()
    vector = model.encode(query).tolist()
    res = client.query(f"""
        SELECT name, brand, category, price_inr, stock_qty, warranty_months, rating, in_stock, price_tier, page_content, cosineDistance(embedding, {vector}) as dist
        FROM product_catalog
        ORDER BY dist ASC LIMIT 2
    """)
    out = []
    if res.result_rows:
        for r in res.result_rows:
            out.append(f"Product: {r[0]} ({r[1]} {r[2]}), Price: {r[3]}, Stock: {r[4]}, Warranty: {r[5]}mo, Rating: {r[6]}, Tier: {r[8]}, Details: {r[9]}")
    return "\n".join(out)

def get_order_context(customer_id: str) -> str:
    client = get_client()
    res = client.query(f"SELECT order_id, order_date, status, items, final_amount_inr FROM order_history WHERE customer_id = '{customer_id}' ORDER BY order_date DESC LIMIT 3")
    out = []
    if res.result_rows:
        for r in res.result_rows:
            out.append(f"Order: {r[0]}, Date: {r[1]}, Status: {r[2]}, Total: {r[4]}, Items: {r[3]}")
    return "\n".join(out)

def get_history_context(customer_id: str) -> str:
    client = get_client()
    res = client.query(f"SELECT ticket_id, call_start_time, summary FROM call_tickets WHERE customer_id = '{customer_id}' ORDER BY call_start_time DESC LIMIT 3")
    out = []
    if res.result_rows:
        for r in res.result_rows:
            out.append(f"Ticket: {r[0]}, Date: {r[1]}, Summary: {r[2]}")
    return "\n".join(out)