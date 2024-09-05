

def find_langchain_model(platform, model_id, **model_kwargs):
    if platform.lower() == "replicate":
        from langchain_community.llms import Replicate
        model = Replicate(
            model=model_id,
            model_kwargs=model_kwargs
        )
    elif platform.lower() == "ollama":
        from langchain_ollama.llms import OllamaLLM
        model = OllamaLLM(model=model_id, **model_kwargs)
    return model
