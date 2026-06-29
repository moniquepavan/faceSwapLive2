# Face Swap Live

Troca de rosto em tempo real via webcam, direto no navegador.

**Stack:** Python · Flask · InsightFace · ONNX Runtime · OpenCV

Funciona em **Windows** e **macOS** (Apple Silicon).

---

## Requisitos

- **Python 3.10, 3.11 ou 3.12** — [download](https://www.python.org/downloads/)
- Webcam
- ~300 MB de espaço (modelos de IA baixados automaticamente na primeira execução)

---

## Windows

### Instalação

**1. Clone o repositório**
```bash
git clone https://github.com/moniquepavan/faceSwapLive2.git
cd faceSwapLive2
```

**2. Dê duplo clique em `instalar.bat`**
Instala as dependências e baixa os modelos de IA automaticamente (~280 MB).

### Uso

Dê duplo clique em **`rodar.bat`**. O navegador abre sozinho em `http://localhost:5000`.

> 💡 GPU NVIDIA acelera muito (15–30 FPS). Sem ela, roda via GPU integrada ou CPU.

---

## macOS (Apple Silicon — M1/M2/M3)

No Mac a instalação é manual (o `.bat` é só Windows). São 4 comandos.

### Instalação

**1. Instale o Python 3.12** (caso não tenha — via [Homebrew](https://brew.sh))
```bash
brew install python@3.12
```

**2. Clone o repositório**
```bash
git clone https://github.com/moniquepavan/faceSwapLive2.git
cd faceSwapLive2
```

**3. Crie o ambiente virtual e instale as dependências**

> ⚠️ No Mac **não** se instala o `onnxruntime-directml` (é exclusivo do Windows).
> Por isso instalamos os pacotes diretamente, sem o `requirements.txt`.

```bash
python3.12 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install flask opencv-python numpy insightface onnxruntime huggingface_hub pillow
```

### Uso

```bash
PORT=5001 ./venv/bin/python app.py
```

Depois abra **`http://localhost:5001`** no navegador.

> **Por que `PORT=5001`?** No macOS, a porta 5000 é ocupada pelo **AirPlay Receiver**.
> Use outra porta (ex.: 5001) ou desative o AirPlay em
> *Ajustes do Sistema → Geral → AirDrop e Handoff → Receptor do AirPlay*.

Na **primeira vez** que clicar em *Iniciar Câmera*, o macOS pede **permissão de câmera** — autorize para o Terminal.

---

## Como usar (Windows e Mac)

1. Faça **upload de uma foto** com o rosto que quer aplicar
2. Clique em **Iniciar Câmera** (autorize a câmera no navegador/sistema)
3. Veja a troca acontecer ao vivo

---

## Performance

A aceleração é detectada automaticamente — não precisa configurar nada.

| Hardware | Aceleração | FPS aprox. |
|---|---|---|
| NVIDIA RTX (Windows) | CUDA | ~30 |
| NVIDIA GTX (Windows) | CUDA | ~15 |
| Apple M1 Pro / M2 / M3 (Mac) | CoreML (MLProgram + GPU) | ~10 |
| GPU integrada Intel/AMD (Windows) | DirectML | ~1–5 |
| Sem GPU | CPU | ~0.5–1 |

> O display sempre roda a ~30 FPS; o número acima é a taxa de **troca de rosto**.

---

## Solução de problemas

| Problema | Causa / solução |
|---|---|
| **`Port 5000 is in use`** (Mac) | AirPlay Receiver. Rode com `PORT=5001` ou desative o AirPlay (ver acima). |
| **Câmera não abre** (Mac) | Autorize a câmera para o Terminal em *Ajustes → Privacidade e Segurança → Câmera*. |
| **Trava / FPS muito baixo no Mac** | Confirme que o status mostra `[Apple GPU (CoreML)]`. Se mostrar `[CPU]`, reinstale o `onnxruntime`. |
| **Erro de modelo faltando** | Os modelos (`inswapper_128`, `buffalo_l`, `buffalo_sc`) baixam sozinhos na 1ª execução. Precisa de internet. |
| **Nenhum rosto detectado** | Use uma foto nítida, de frente, com boa iluminação. |

---

## Como funciona (resumo técnico)

- **Captura** roda numa thread; **detecção + swap** em outra; **streaming MJPEG** numa terceira. Assim o vídeo nunca congela mesmo quando a troca está pesada.
- A **detecção** de rosto (modelo leve `buffalo_sc`) roda 1 a cada 6 frames e o resultado é cacheado; o **swap** (`inswapper_128`) roda a cada frame.
- O provider de inferência é escolhido automaticamente por plataforma:
  CUDA (NVIDIA) → DirectML (Windows) → CoreML/MLProgram (Apple Silicon) → CPU.
