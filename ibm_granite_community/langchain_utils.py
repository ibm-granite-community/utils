from ibm_granite_community.notebook_utils import set_api_key

def find_langchain_model(platform, model_id, **model_kwargs):
    if platform.lower() == "replicate":
        from langchain_community.llms import Replicate
        set_api_key()
        model = Replicate(
            model=model_id,
            model_kwargs=model_kwargs
        )
    elif platform.lower() == "ollama":
        from langchain_ollama.llms import OllamaLLM
        model = OllamaLLM(model=model_id, **model_kwargs)
    return model


def find_langchain_vector_db(provider):
    if provider == "milvus":
        from langchain_milvus import Milvus
        
        # Create a local Milvus db
        db_file = "/tmp/milvus_for_rag.db"
        return Milvus(embedding_function=embeddings_model, connection_args={"uri": db_file}, auto_id=True)

    elif provider == "chroma":
        from langchain_chroma import Chroma
        return Chroma(embedding_function=embeddings_model)

    else:
        raise ValueError(f"Invalid vector store provider '{provider}'")