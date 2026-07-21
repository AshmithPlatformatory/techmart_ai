import json
import os
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class ProductSpecs(BaseModel):
    product_id: str
    ram_gb: Optional[str] = Field(None, description="RAM in GB")
    storage_gb: Optional[str] = Field(None, description="Storage in GB (e.g. SSD, HDD, ROM)")
    battery: Optional[str] = Field(None, description="Battery life in hours or mAh")
    screen_inches: Optional[str] = Field(None, description="Screen size in inches")
    resolution: Optional[str] = Field(None, description="Screen or Video resolution (e.g. 4K, 1080p, 1440p)")
    refresh_rate_hz: Optional[str] = Field(None, description="Refresh rate in Hz")
    camera_mp: Optional[str] = Field(None, description="Camera megapixels")
    cpu: Optional[str] = Field(None, description="Processor or CPU model")
    gpu: Optional[str] = Field(None, description="Graphics or GPU model")
    os: Optional[str] = Field(None, description="Operating System")
    display_type: Optional[str] = Field(None, description="Display tech (e.g. OLED, LCD, Retina)")
    anc: Optional[str] = Field(None, description="Active Noise Cancellation features")
    connectivity: Optional[str] = Field(None, description="Connectivity (e.g. Bluetooth, USB-C, Wi-Fi)")
    features: Optional[str] = Field(None, description="Any other standout hardware features")

class BatchSpecsOutput(BaseModel):
    products: List[ProductSpecs]

def migrate_catalog(file_path):
    print(f"Reading {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} products. Cleaning old specs...")
    for product in data:
        if "specs" in product.get("metadata", {}):
            del product["metadata"]["specs"]
            
    # Save the cleaned state immediately to avoid confusion
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(BatchSpecsOutput)

    batch_size = 9
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    print(f"Starting LLM extraction in {total_batches} batches...")
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        prompt = "Extract technical specifications for the following products. ONLY extract hardware/technical specs. DO NOT extract price, rating, stock, or warranty.\n\n"
        for p in batch:
            prompt += f"ID: {p['product_id']}\nDescription: {p['page_content']}\n\n"
            
        try:
            result = structured_llm.invoke(prompt)
            # Map results back
            spec_map = {item.product_id: {k: str(v) for k, v in item.dict().items() if v is not None and k != "product_id"} for item in result.products}
            for p in batch:
                extracted_specs = spec_map.get(p["product_id"], {})
                p["metadata"]["specs"] = extracted_specs
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            
        time.sleep(1) # Small delay to avoid rate limits
        
    print(f"Saving to {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print("LLM Migration complete.")

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "product_catalog_rag.json")
    migrate_catalog(path)
