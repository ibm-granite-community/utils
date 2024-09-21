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

    # Uncomment Pinecone when it can be tested by adding PINECONE_API_KEY to github CI
    # elif provider == "pinecone":
    #     from langchain_pinecone import PineconeVectorStore
    #     from pinecone import Pinecone

    #     vector_db_class = PineconeVectorStore
    #     pc = Pinecone(api_key=get_env_var("PINECONE_API_KEY"))

    #     # This index must already exist in your Pinecone account. The dimensions (length) of the index vector should match the embedding model's embedding dimension.
    #     index_name = "rag-recipe-example"
    #     index = pc.Index(index_name)

    #     return PineconeVectorStore(index=index, embedding=embeddings_model)

    else:
        raise ValueError(f"Invalid vector store provider '{provider}'")