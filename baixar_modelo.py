"""
Baixa o modelo inswapper_128.onnx do Hugging Face.
Execute: python baixar_modelo.py (ou venv\Scripts\python baixar_modelo.py)
"""
import os, sys, shutil

DEST = os.path.join(os.path.dirname(__file__), "models", "inswapper_128.onnx")


def main():
    if os.path.exists(DEST):
        print(f"Modelo já existe: {DEST}  ({os.path.getsize(DEST)/1e6:.0f} MB)")
        return

    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Instalando huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub --quiet")
        from huggingface_hub import hf_hub_download

    print("=" * 55)
    print(" Download do modelo de face swap (~280 MB)")
    print("=" * 55)
    print()
    print("Precisa de uma conta GRATUITA no Hugging Face:")
    print("  1. Crie em: https://huggingface.co")
    print("  2. Gere token em: https://huggingface.co/settings/tokens")
    print("     (New token -> Read -> Create -> Copie o token)")
    print()
    token = input("Cole seu token HF aqui: ").strip()
    if not token:
        print("Token nao informado. Abortando.")
        sys.exit(1)

    print("\nBaixando... (pode demorar alguns minutos)")
    try:
        path = hf_hub_download(
            repo_id="hacksider/deep-live-cam",
            filename="inswapper_128_fp16.onnx",
            token=token,
            local_dir=os.path.dirname(DEST),
        )
        if os.path.abspath(path) != os.path.abspath(DEST):
            shutil.move(path, DEST)

        print(f"\nDownload concluido! {os.path.getsize(DEST)/1e6:.0f} MB")
        print(f"Salvo em: {DEST}")
    except Exception as e:
        print(f"\nErro no download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
