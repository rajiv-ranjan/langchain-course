# Notes

## Ollama — List Available Models

To list all locally available models in Ollama:

```bash
ollama list
```

To see detailed info for a specific model:

```bash
ollama show <model-name>
```

---

## Locally Installed Ollama Models

| Model | Parameters | Context Window | Quantization | Tool Calling | Vision | Thinking | Size on Disk |
|---|---|---|---|---|---|---|---|
| `llama3.3:latest` | 70.6B | 131,072 | Q4_K_M | Yes | No | No | 42 GB |
| `llama3.2:3b` | 3.2B | 131,072 | Q4_K_M | Yes | No | No | 2.0 GB |
| `gemma4:latest` | 8.0B | 131,072 | Q4_K_M | Yes | Yes | Yes | 9.6 GB |
| `qwen3.5:0.8b` | 873M | 262,144 | Q8_0 | Yes | Yes | Yes | 1.0 GB |
| `qwen3.5:27b` | 27.8B | 262,144 | Q4_K_M | Yes | Yes | Yes | 17 GB |
| `gpt-oss:latest` | 20.9B | 131,072 | MXFP4 | Yes | No | Yes | 13 GB |
| `gemma3:12b` | 12.2B | 131,072 | Q4_K_M | No | Yes | No | 8.1 GB |
| `functiongemma:latest` | 268M | 32,768 | Q8_0 | Yes | No | No | 300 MB |
| `llama3.1:8b` | 8.0B | 131,072 | Q4_K_M | Yes | No | No | 4.9 GB |
| `llava-llama3:8b` | 8B | 8,192 | Q4_K_M | No | Yes | No | 5.5 GB |

### Notes on Quantization
- **Q4_K_M** — 4-bit quantization (compressed, reduced size/precision)
- **Q8_0** — 8-bit quantization (compressed, higher precision than Q4)
- **MXFP4** — Microsoft MX Float 4-bit quantization (compressed)

All models listed are quantized (compressed) versions. None are full-precision (FP16/BF16/FP32).
