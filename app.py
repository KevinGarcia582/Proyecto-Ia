"""Aplicación Streamlit de análisis de reseñas. Ejecutar con streamlit run app.py."""
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from src.agent import build_agent
from src.model import predict_text, model_status, load_model_resource

load_dotenv()
st.set_page_config(page_title="Reseñas con criterio", page_icon="📝", layout="wide")
st.title("📝 Analizador de reseñas y triaje")
st.caption("Ruta 3 · PLN con embeddings + agente con herramientas")
with st.expander("Arquitectura del sistema", expanded=True):
    st.graphviz_chart('''digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fontname="sans-serif"];
      UI [label="Streamlit", fillcolor="#dbeafe"];
      AG [label="Agente LangChain / Gemini", fillcolor="#ede9fe"];
      MEM [label="Memoria por thread_id", shape=cylinder, fillcolor="#e2e8f0"];
      T1 [label="@tool 1 · Inferencia U4", fillcolor="#fee2e2"];
      NN [label="Red seleccionada en validación", fillcolor="#fecaca"];
      T2 [label="@tool 2 · Reglas de triaje", fillcolor="#dcfce7"];
      UI -> AG; AG -> MEM [dir=both]; AG -> T1; T1 -> NN; AG -> T2;
    }''')

@st.cache_resource(show_spinner="Cargando la red neuronal…")
def load_predictor():
    status = model_status()
    if status["ready"]:
        load_model_resource()
    return status

@st.cache_resource(show_spinner="Preparando el agente…")
def get_agent(api_key: str):
    return build_agent()

status = load_predictor()
if not status["ready"]:
    st.warning("Aún no hay pesos entrenados. Ejecuta notebooks/entrenamiento_u4.ipynb y guarda el modelo en models/modelo_seleccionado.keras.")
else:
    st.success(f"Modelo listo: {status['name']}")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Sesión")
    st.code(st.session_state.thread_id, language=None)
    if st.button("Reiniciar conversación"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("La clave GOOGLE_API_KEY se configura en .env local. Nunca la publiques.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Predicción directa (U4)")
    review = st.text_area("Escribe una reseña en inglés (formato IMDB)", key="review", height=150)
    if st.button("Clasificar reseña", disabled=not status["ready"]):
        if not review.strip():
            st.warning("Escribe una reseña antes de clasificarla.")
        else:
            result = predict_text(review)
            st.write(f"**Sentimiento:** {result['label']}")
            st.write(f"**Confianza:** {result['confidence']:.1%}")
            st.json(result)
with col2:
    st.subheader("Conversación con el agente (U5)")
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("tools"):
                st.caption("Herramientas invocadas: " + ", ".join(item["tools"]))
    prompt = st.chat_input("Pregunta sobre una reseña o las reglas de triaje")
    if prompt:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Configura GOOGLE_API_KEY en .env y reinicia la aplicación.")
        else:
            st.session_state.messages.append({"role":"user", "content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Consultando al agente…"):
                    try:
                        agent = get_agent(os.getenv("GOOGLE_API_KEY", ""))
                        response = agent.invoke(
                            {"messages":[{"role":"user", "content":prompt}]},
                            config={"configurable":{"thread_id":st.session_state.thread_id}},
                        )
                        answer = response["messages"][-1].content
                        tool_names = [call["name"] for msg in response["messages"] for call in (getattr(msg,"tool_calls",[]) or [])]
                        st.markdown(answer)
                        if tool_names:
                            st.caption("Herramientas invocadas: " + ", ".join(tool_names))
                        st.session_state.messages.append({"role":"assistant", "content":answer, "tools":tool_names})
                    except Exception as exc:
                        st.error(f"No se pudo consultar el agente: {exc}")
