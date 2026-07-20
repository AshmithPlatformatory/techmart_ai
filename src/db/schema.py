from src.db.clickhouse_client import get_client
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import argparse

def create_tables(reset=False):
    client = get_client()

    if reset:
        tables_to_drop = [
            "customers", "order_history", "company_faqs", 
            "company_tos", "product_catalog", "call_tickets"
        ]
        for table in tables_to_drop:
            client.command(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"Dropped existing table: {table}")

    schema_statements = [
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id String,
            name String,
            email String,
            phone String,
            age Int32,
            gender String,
            address String,
            city String,
            state String,
            pincode String,
            account_created_at String,
            loyalty_tier String,
            preferred_payment String,
            is_active UInt8
        ) ENGINE = MergeTree()
        ORDER BY customer_id
        """,
        """
        CREATE TABLE IF NOT EXISTS order_history (
            order_id String,
            customer_id String,
            order_date String,
            status String,
            payment_method String,
            delivery_address String,
            items String,
            subtotal_inr Float32,
            discount_inr Float32,
            delivery_charges_inr Float32,
            final_amount_inr Float32,
            item_count Int32
        ) ENGINE = MergeTree()
        ORDER BY order_id
        """,
        """
        CREATE TABLE IF NOT EXISTS company_faqs (
            faq_id String,
            category String,
            question String,
            answer String,
            tags Array(String),
            related_tos String,
            question_embedding Array(Float32)
        ) ENGINE = MergeTree()
        ORDER BY faq_id
        """,
        """
        CREATE TABLE IF NOT EXISTS company_tos (
            doc_id String,
            title String,
            category String,
            tags Array(String),
            effective_date Date,
            last_updated Date,
            content String
        ) ENGINE = MergeTree()
        ORDER BY doc_id
        """,
        """
        CREATE TABLE IF NOT EXISTS product_catalog (
            product_id String,
            page_content String,
            name String,
            brand String,
            category String,
            price_inr Float32,
            stock_qty Int32,
            warranty_months Int32,
            rating Float32,
            in_stock UInt8,
            price_tier String,
            embedding Array(Float32)
        ) ENGINE = MergeTree()
        ORDER BY product_id
        """,
        """
        CREATE TABLE IF NOT EXISTS call_tickets (
            ticket_id String,
            session_id String,
            customer_id String,
            caller_phone String,
            call_start_time DateTime,
            call_end_time Nullable(DateTime),
            call_status String,
            full_transcript String,
            summary String,
            summary_embedding Array(Float32)
        ) ENGINE = MergeTree()
        ORDER BY ticket_id
        """
    ]

    for statement in schema_statements:
        client.command(statement)
        
    logger.info("ClickHouse schema deployment complete.")

def create_indexes():
    client = get_client()
    logger.info("Creating HNSW vector indexes...")
    
    try:
        client.command("SET allow_experimental_vector_similarity_index = 1")
    except Exception as e:
        logger.info(f"SET allow_experimental_vector_similarity_index failed (may not be needed): {e}")

    indexes = [
        """
        ALTER TABLE company_faqs 
        ADD INDEX IF NOT EXISTS hnsw_faq_idx question_embedding 
        TYPE vector_similarity('hnsw', 'cosineDistance', 384, 'f32', 100, 500)
        """,
        """
        ALTER TABLE product_catalog 
        ADD INDEX IF NOT EXISTS hnsw_catalog_idx embedding 
        TYPE vector_similarity('hnsw', 'cosineDistance', 384, 'f32', 100, 500)
        """,
        """
        ALTER TABLE call_tickets 
        ADD INDEX IF NOT EXISTS hnsw_tickets_idx summary_embedding 
        TYPE vector_similarity('hnsw', 'cosineDistance', 384, 'f32', 100, 500)
        """
    ]
    
    for stmt in indexes:
        try:
            client.command(stmt)
        except Exception as e:
            logger.info(f"Index creation skipped/failed: {e}")
            
    logger.info("Materializing indexes...")
    materialize_stmts = [
        "ALTER TABLE company_faqs MATERIALIZE INDEX IF EXISTS hnsw_faq_idx",
        "ALTER TABLE product_catalog MATERIALIZE INDEX IF EXISTS hnsw_catalog_idx",
        "ALTER TABLE call_tickets MATERIALIZE INDEX IF EXISTS hnsw_tickets_idx"
    ]
    
    for stmt in materialize_stmts:
        try:
            client.command(stmt)
        except Exception as e:
            logger.info(f"Index materialization skipped/failed: {e}")

    logger.info("HNSW index setup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    parser.add_argument("--add-indexes", action="store_true", help="Create HNSW indexes on existing tables")
    args = parser.parse_args()
    if args.add_indexes:
        create_indexes()
    else:
        create_tables(reset=args.reset)