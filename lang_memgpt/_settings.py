from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: str = "fd9453a9-749a-4400-9673-053edbbe70a7"
    pinecone_index_name: str = "membot"
    pinecone_namespace: str = "Vought"
    model: str = "gpt-3.5-turbo"


SETTINGS = Settings()
