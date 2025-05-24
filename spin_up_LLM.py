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

def spin_up_LLM(chosen_llm: str, local_or_remote: str = "local"):
    """
    Spins up and returns a LangChain-compatible LLM.
    
    Args:
      chosen_llm:  name of the model to load (e.g. "mistral", "gemma3")
      local_or_remote:  "local" to use Ollama; otherwise a NotImplementedError is raised.
    
    Returns:
      A langchain_ollama.llms.OllamaLLM instance.
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
        print(f"Available models:\n{lst.stdout}")

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
        return OllamaLLM(model=chosen_llm)

    else:
        # Placeholder for remote LLM logic
        raise NotImplementedError("Remote LLM support is not yet implemented.")
