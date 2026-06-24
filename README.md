# Face Swap Live

Aplicação web de **troca de rosto em tempo real via webcam**. Faça upload de uma foto com o rosto desejado e veja a substituição acontecer ao vivo no navegador.

**Stack:** Python · Flask · InsightFace · ONNX Runtime · OpenCV

---

## Pré-requisitos

- Python **3.10, 3.11 ou 3.12** ([download](https://www.python.org/downloads/)) — Python 3.13 também funciona
- Webcam conectada
- Windows 10/11 (testado), macOS ou Linux
- Conta gratuita no [Hugging Face](https://huggingface.co) para baixar o modelo de IA

> **GPU NVIDIA (opcional):** se tiver GPU NVIDIA com CUDA, o swap roda a ~15–30 FPS. Sem GPU dedicada, roda a ~1 FPS usando a GPU integrada via DirectML.

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/moniquepavan/faceSwapLive2.git
cd faceSwapLive2
```

### 2. Crie o ambiente virtual e instale as dependências

**Windows:**
```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### 3. Baixe o modelo de IA

O modelo de troca de rosto (~280 MB) precisa ser baixado uma única vez.

**3.1** Crie uma conta gratuita em [huggingface.co](https://huggingface.co)

**3.2** Gere um token de acesso em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
→ clique em **New token** → tipo **Read** → copie o token (começa com `hf_...`)

**3.3** Execute o script de download:

```bash
# Windows
venv\Scripts\python baixar_modelo.py

# macOS / Linux
venv/bin/python baixar_modelo.py
```

Cole o token quando solicitado. O download leva alguns minutos dependendo da sua internet.

---

## Como rodar

**Windows:**
```bash
venv\Scripts\python app.py
```

**macOS / Linux:**
```bash
venv/bin/python app.py
```

Abra o navegador em **http://localhost:5000**

---

## Como usar

1. Acesse `http://localhost:5000`
2. **Arraste ou clique** na área de upload para enviar uma foto com o rosto que deseja aplicar
3. Clique em **Iniciar Câmera**
4. Seu rosto na webcam será substituído pelo rosto da foto em tempo real

---

## Estrutura do projeto

```
faceSwapLive2/
├── app.py            # Servidor Flask + streaming MJPEG
├── face_swap.py      # Pipeline de detecção e swap com InsightFace
├── baixar_modelo.py  # Script de download do modelo
├── requirements.txt  # Dependências Python
├── templates/
│   └── index.html    # Interface web
└── models/           # Pasta onde o modelo .onnx é salvo (não versionada)
```

---

## Observações de performance

| Hardware | FPS do swap |
|---|---|
| CPU apenas | ~0.5 FPS |
| GPU integrada Intel (DirectML) | ~1 FPS |
| GPU NVIDIA GTX 1060 | ~15 FPS |
| GPU NVIDIA RTX série | ~30–60 FPS |

O vídeo da webcam sempre roda a ~26 FPS independente do hardware. O swap acontece em background e atualiza o rosto conforme a velocidade do hardware permite.

Se tiver GPU NVIDIA, instale o `onnxruntime-gpu` no lugar do `onnxruntime`:
```bash
venv\Scripts\pip uninstall onnxruntime-directml
venv\Scripts\pip install onnxruntime-gpu
```
E altere `_SWAP_PROVIDERS` em `face_swap.py` para `["CUDAExecutionProvider", "CPUExecutionProvider"]`.
