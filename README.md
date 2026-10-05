# llm-chat-cli

Herramienta de línea de comandos para chatear con LLMs (Anthropic, OpenAI y Groq) usando únicamente los SDKs oficiales, sin frameworks como LangChain.

## Descripción

Un chat en la terminal que:

- mantiene una **conversación multi-turno** (el historial se maneja a mano, con una ventana deslizante para no pasarse del contexto),
- muestra la respuesta con **streaming**, token por token,
- cambia de proveedor con un flag: `--model anthropic|openai|groq`,
- devuelve **salidas estructuradas validadas con Pydantic** (comando `/summary`), reintentando si el modelo responde un JSON inválido,
- muestra mensajes de error claros (clave inválida, rate limit, sin conexión) y reintenta con backoff exponencial.

**Stack:** Python 3.11+, SDKs oficiales de `anthropic` y `openai`, `pydantic`, `python-dotenv`, `pytest`.

## Instalación

```bash
git clone <url-del-repo>
cd llm-chat-cli

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # y completá al menos una API key
```

Variables de `.env`:

| Variable | Para qué |
|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY` | credenciales (solo hace falta la del proveedor que uses) |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GROQ_MODEL` | modelo a usar en cada proveedor |
| `GROQ_BASE_URL` | endpoint de Groq (compatible con OpenAI) |

> Los nombres de modelos cambian seguido. Si ves "Model not found", revisá el `*_MODEL` de tu `.env`.

## Uso

```bash
llm-chat-cli --model groq
llm-chat-cli --model openai --max-history 10
```

| Flag | Descripción |
|---|---|
| `--model` | `anthropic` (por defecto), `openai` o `groq` |
| `--max-history` | máximo de mensajes que se recuerdan; los más viejos se descartan (por defecto 20) |

Comandos dentro del chat:

| Comando | Qué hace |
|---|---|
| `/summary <texto>` | resume el texto como JSON validado con Pydantic (título, puntos clave, sentimiento) |
| `exit`, `quit`, Ctrl+C | salir |

### Demo

Sesión real con Groq:

```text
you> Me llamo Sole y estoy aprendiendo a programar con LLMs.
bot> ¡Hola Sole! 👋 Qué bueno que estés empezando a programar con LLMs. ¿En qué lenguaje
     o framework te estás enfocando?

you> ¿Cómo me llamo y qué estoy aprendiendo? Respondé en una frase.
bot> Te llamas Sole y estás aprendiendo a programar con LLMs.

you> /summary El nuevo restaurante del barrio tiene platos ricos y precios justos, aunque el servicio fue lento y tuvimos que esperar mucho.
{
  "title": "Resumen",
  "key_points": [
    "El nuevo restaurante del barrio ofrece platos ricos y precios justos",
    "El servicio fue lento y se tuvo que esperar mucho"
  ],
  "sentiment": "neutral"
}
```

### Tests

```bash
pytest
```

Los tests no usan red ni claves: emplean un proveedor falso y excepciones construidas a mano.

## Arquitectura

```
src/llm_chat_cli/
├── cli.py                  # argumentos, loop de chat, historial
├── models.py               # Message, Summary, trim_history
├── prompts.py              # system prompts
├── structured.py           # ask_structured: JSON + validación + reintento
└── providers/
    ├── base.py             # interfaz Provider: stream(messages, system)
    ├── anthropic_provider.py
    ├── openai_provider.py  # sirve para OpenAI y Groq
    ├── errors.py           # excepciones de los SDKs -> mensajes claros
    └── __init__.py         # create_provider(name): la factory
```

La idea central: `cli.py` no sabe con qué proveedor habla. Solo usa la interfaz `Provider.stream()`, que devuelve el texto en trozos a medida que llega, y cada proveedor traduce a su SDK. Para agregar uno nuevo alcanza con implementar esa interfaz y registrarlo en la factory.

Groq reutiliza el adaptador de OpenAI (`OpenAIProvider`), ya que su API es compatible con la de OpenAI: solo cambian la clave, el modelo y la `base_url`.

La memoria es una lista de `Message` que se reenvía completa en cada turno, porque las APIs no guardan estado. `trim_history` la recorta como ventana deslizante y nunca deja un mensaje `assistant` al principio, ya que Anthropic exige que la conversación empiece con `user`.

## Diferencias entre proveedores

| | Anthropic | OpenAI | Groq |
|---|---|---|---|
| System prompt | parámetro `system` aparte | mensaje con `role: "system"` dentro de `messages` | igual que OpenAI |
| `max_tokens` | obligatorio | opcional | opcional |
| Streaming | `client.messages.stream()` + `text_stream` | `create(stream=True)` e iterar chunks | igual que OpenAI |
| Texto de la respuesta | `response.content[0].text` | `response.choices[0].message.content` | igual que OpenAI |
| Texto de cada chunk | ya viene como `str` | `chunk.choices[0].delta.content` (puede ser `None`) | igual que OpenAI |
| Primer mensaje | debe ser `user` | sin esa restricción | igual que OpenAI |
| Cliente en este proyecto | `AnthropicProvider` | `OpenAIProvider` | `OpenAIProvider` con otra `base_url` |

## Estado

Probado con la API real de **Groq**. `AnthropicProvider` y el uso con OpenAI todavía no se probaron contra sus APIs reales (solo con tests y un proveedor falso).

## Posibles mejoras

- Recortar el historial por tokens o resumir lo viejo, en vez de contar mensajes.
- Circuit breaker para no insistir cuando el servicio está caído.
- Guardar y retomar conversaciones.
