"""
Baixa o modelo inswapper_128.onnx automaticamente.
Tenta sem token (repo publico). Se falhar, pede token do Hugging Face.
"""
import os, sys, shutil

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "inswapper_128.onnx")

SOURCES = [
    # repo publico, sem token
    dict(repo_id="hacksider/deep-live-cam", filename="inswapper_128_fp16.onnx", token=None),
    dict(repo_id="deepinsight/inswapper",   filename="inswapper_128.onnx",      token=None),
]


def _ensure_buffalo_sc(silent=False):
    """Baixa o detector buffalo_sc (det_500m.onnx) se faltar.
    O FaceAnalysis baixa o buffalo_l sozinho, mas o buffalo_sc e carregado
    por caminho direto em face_swap.py e precisa ser baixado a parte."""
    det = os.path.join(os.path.expanduser("~"), ".insightface", "models",
                       "buffalo_sc", "det_500m.onnx")
    if os.path.exists(det):
        return
    if not silent:
        print("Baixando detector buffalo_sc (~15 MB)...")
    from insightface.utils import storage
    storage.ensure_available("models", "buffalo_sc")


def main(silent=False):
    _ensure_buffalo_sc(silent)

    if os.path.exists(DEST) and os.path.getsize(DEST) > 50_000_000:
        if not silent:
            print(f"Modelo ja existe ({os.path.getsize(DEST)/1e6:.0f} MB). OK!")
        return True

    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        os.system(f'"{sys.executable}" -m pip install huggingface_hub --quiet')
        from huggingface_hub import hf_hub_download

    print("=" * 55)
    print(" Baixando modelo de face swap (~280 MB)")
    print("=" * 55)

    # Tenta fontes publicas sem token
    for src in SOURCES:
        try:
            print(f"\nTentando {src['repo_id']}...")
            path = hf_hub_download(
                repo_id=src["repo_id"],
                filename=src["filename"],
                token=src["token"],
                local_dir=os.path.dirname(DEST),
            )
            _mover(path)
            return True
        except Exception as e:
            print(f"  Falhou: {e}")

    # Pede token como ultimo recurso
    print()
    print("Download publico falhou. Precisa de um token do Hugging Face.")
    print("  1. Crie conta gratuita em: https://huggingface.co")
    print("  2. Gere token em:          https://huggingface.co/settings/tokens")
    print()
    token = input("Cole seu token aqui (ou Enter para cancelar): ").strip()
    if not token:
        print("Cancelado.")
        return False

    try:
        path = hf_hub_download(
            repo_id="hacksider/deep-live-cam",
            filename="inswapper_128_fp16.onnx",
            token=token,
            local_dir=os.path.dirname(DEST),
        )
        _mover(path)
        return True
    except Exception as e:
        print(f"Erro: {e}")
        return False


def _mover(path):
    if os.path.abspath(path) != os.path.abspath(DEST):
        shutil.move(path, DEST)
    print(f"\nModelo salvo! ({os.path.getsize(DEST)/1e6:.0f} MB)")


if __name__ == "__main__":
    ok = main()
    if not ok:
        sys.exit(1)
