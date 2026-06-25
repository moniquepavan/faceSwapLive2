# Face Swap Live

Troca de rosto em tempo real via webcam no navegador.

**Stack:** Python · Flask · InsightFace · ONNX Runtime · OpenCV

---

## Requisitos

- Python 3.10, 3.11 ou 3.12 — [download](https://www.python.org/downloads/) (marque "Add Python to PATH")
- Webcam
- Windows 10/11

> GPU NVIDIA melhora muito o desempenho (15–30 FPS). Sem ela roda em ~1 FPS via GPU integrada.

---

## Instalação e uso (Windows)

**1. Clone o repositório**
```bash
git clone https://github.com/moniquepavan/faceSwapLive2.git
cd faceSwapLive2
```

**2. Dê duplo clique em `instalar.bat`**
Instala as dependências e baixa o modelo de IA automaticamente (~280 MB).

**3. Dê duplo clique em `rodar.bat`**
O navegador abre sozinho em `http://localhost:5000`.

---

## Como usar

1. Faça upload de uma foto com o rosto que quer aplicar
2. Clique em **Iniciar Câmera**
3. Veja a troca acontecer ao vivo

---

## Performance

| Hardware | FPS |
|---|---|
| CPU | ~0.5 FPS |
| GPU integrada Intel/AMD | ~1 FPS |
| GPU NVIDIA GTX | ~15 FPS |
| GPU NVIDIA RTX | ~30 FPS |

O código detecta automaticamente a melhor GPU disponível — não precisa configurar nada.
