import shutil
import subprocess
from time import sleep
import requests


def wait_for_ollama_ready(host: str = "127.0.0.1", port: int = 11434, timeout: int = 15):
    url = f"http://{host}:{port}"
    for _ in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            sleep(1)
    raise RuntimeError(f"Ollama server at {url} failed to start within {timeout}s.")

def spin_up_LLM(
    chosen_llm: str,
    local_or_remote: str = "local",
    provider: str = None,
    api_key: str = None,
    **kwargs
):
    """
    Spins up and returns a LangChain-compatible LLM for local (Ollama) or remote (OpenAI) use.
    
    Args:
      chosen_llm:      name of the model to load (e.g. "mistral", "gemma3", or "gpt-4o")
      local_or_remote: "local" for Ollama, "remote" for API (OpenAI etc)
      provider:        "openai" for remote usage
      api_key:         API key for the remote provider (if needed)
      kwargs:          Any additional params passed to the LLM initializer
    
    Returns:
      A LangChain LLM object (OllamaLLM or ChatOpenAI)
    """
    if local_or_remote == "local":
        # 1) Install Ollama if missing
        if shutil.which('ollama') is None:
            print("🚀 Installing Ollama...")
            install = subprocess.run(
                'curl https://ollama.ai/install.sh | sh',
                capture_output=True, text=True, shell=True
            )
            if install.returncode != 0:
                raise RuntimeError(f"Error installing Ollama: {install.stderr}")

        # 2) Start Ollama server
        print("🚀 Starting Ollama server...")
        serve_cmd = (
            'OLLAMA_HOST=127.0.0.1:11434 '
            'ollama serve > serve.log 2>&1 &'
        )
        proc = subprocess.Popen(
            serve_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        print(f"→ Ollama PID: {proc.pid}")

        # 3) Wait until it’s ready
        print("⏳ Waiting for Ollama to be ready…")
        wait_for_ollama_ready()
        print("Ready!\n")

        # 4) Pull the requested model
        print(f"🚀 Pulling model '{chosen_llm}'…")
        pull = subprocess.run(
            f'ollama pull {chosen_llm}',
            capture_output=True, text=True, shell=True
        )
        if pull.returncode != 0:
            raise RuntimeError(f"Error pulling '{chosen_llm}': {pull.stderr}")

        # 5) List available models
        lst = subprocess.run(
            'ollama list',
            capture_output=True, text=True, shell=True
        )
        if lst.returncode != 0:
            raise RuntimeError(f"Error listing models: {lst.stderr}")
        print(f"\nAvailable models:\n{lst.stdout}")

        # 6) Ensure langchain-ollama is installed
        print("🚀 Installing langchain-ollama…")
        pip = subprocess.run(
            'pip install -U langchain-ollama', capture_output=True, text=True, shell=True
        )
        if pip.returncode != 0:
            raise RuntimeError(f"Error installing langchain-ollama: {pip.stderr}")

        # Importing langchain_ollama only here since we installed it above:
        from langchain_ollama.llms import OllamaLLM

        # 7) Return the OllamaLLM wrapper
        print("All done setting up Ollama and local LLM.\n")
        return OllamaLLM(model=chosen_llm)
        
    elif local_or_remote == "remote":
        # Support "openai" (default)
        provider = provider or "openai"
        if provider.lower() == "openai":
            print("🚀 Setting up remote OpenAI model…")
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                print("Installing langchain-openai...")
                pip = subprocess.run(
                    'pip install -U langchain-openai', capture_output=True, text=True, shell=True
                )
                if pip.returncode != 0:
                    raise RuntimeError(f"Error installing langchain-openai: {pip.stderr}")
                from langchain_openai import ChatOpenAI

            # API key required
            import os
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("No OpenAI API key provided. Pass it as 'api_key' or set the OPENAI_API_KEY environment variable.")

            # Return LangChain OpenAI chat model
            print("All done setting up OpenAI LLM.\n")
            return ChatOpenAI(model=chosen_llm, api_key=api_key, **kwargs)
        else:
            raise NotImplementedError(f"Remote provider '{provider}' is not yet supported.")
    else:
        raise ValueError("local_or_remote must be either 'local' or 'remote'.")