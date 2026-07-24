from src.db.clickhouse_client import get_client

client = get_client()
print("Orders for CUST-0221:")
res = client.query("SELECT order_id, status, items, final_amount_inr FROM order_history WHERE customer_id = 'CUST-0221'")
for row in res.result_rows:
    print(row)
