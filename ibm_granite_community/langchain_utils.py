from ibm_granite_community.notebook_utils import set_env_var, get_env_var

def find_langchain_model(platform, model_id, **model_kwargs):
    if platform.lower() == "replicate":
        from langchain_community.llms import Replicate
        set_env_var('REPLICATE_API_TOKEN')
        model = Replicate(
            model=model_id,
            model_kwargs=model_kwargs
        )
    elif platform.lower() == "ollama":
        from langchain_ollama.llms import OllamaLLM
        model = OllamaLLM(model=model_id, **model_kwargs)
    elif platform.lower() == "watsonx":
        from langchain_ibm import WatsonxLLM
        model = WatsonxLLM(
            model_id=model_id, 
            url= get_env_var("WATSONX_URL"),
            apikey=get_env_var("WATSONX_APIKEY"),
            project_id=get_env_var("WATSONX_PROJECT_ID"),
            **model_kwargs
        )
    else:
        raise ValueError(f"Platform {platform} not supported.")
    return model


def find_langchain_vector_db(provider, embeddings_model, **model_kwargs):
    if provider == "milvus":
        from langchain_milvus import Milvus
        return Milvus(embedding_function=embeddings_model, **model_kwargs)

    elif provider == "chroma":
        from langchain_chroma import Chroma
        return Chroma(embedding_function=embeddings_model, **model_kwargs)

    else:
        raise ValueError(f"Invalid vector store provider '{provider}'")