"""Agente LangChain con dos herramientas y memoria por thread_id."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from src.model import predict_text

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
_RULES = ROOT / "data" / "reglas_triaje.md"
_CHECKPOINTER = InMemorySaver()

def _read_rules():
    return _RULES.read_text(encoding="utf-8")

@tool
def clasificar_resena(texto: str) -> str:
    """Clasifica una reseña IMDB en positiva o negativa con la red neuronal local y devuelve confianza. Úsala cuando el usuario entregue texto de una reseña o pida analizar su sentimiento."""
    result = predict_text(texto)
    return f"Predicción: {result['label']}; confianza: {result['confidence']:.3f}; probabilidad positiva: {result['positive_probability']:.3f}."

@tool
def consultar_reglas_triaje(consulta: str) -> str:
    """Busca las reglas del proyecto para interpretar reseñas, incertidumbre y revisión humana. Úsala para preguntas sobre criterios, límites o acciones recomendadas; responde solo con base en el texto recuperado."""
    rules = _read_rules()
    terms = set(consulta.lower().split())
    chunks = [p.strip() for p in rules.split("\n\n") if p.strip()]
    ranked = sorted(chunks, key=lambda p: sum(t in p.lower() for t in terms), reverse=True)
    return "\n\n".join(ranked[:3])

def build_agent():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Falta GOOGLE_API_KEY en el archivo .env")
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"), temperature=0)
    return create_agent(
        model=model,
        tools=[clasificar_resena, consultar_reglas_triaje],
        system_prompt=("Eres un asistente de análisis de reseñas IMDB. Para clasificar texto usa clasificar_resena; "
          "para explicar criterios usa consultar_reglas_triaje. Si preguntan ambas cosas, usa ambas herramientas. "
          "Aclara que una probabilidad no es garantía y evita inventar información."),
        checkpointer=_CHECKPOINTER,
    )
