from fastapi import FastAPI
from src.admin.router import admin_router

app = FastAPI(title="TechMart Admin CMS")

app.include_router(admin_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.admin.app:app", host="0.0.0.0", port=8000, reload=True)