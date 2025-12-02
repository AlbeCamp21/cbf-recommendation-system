import streamlit as st
import sys
import os
from PyPDF2 import PdfReader

# Agregar directorio PLN al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
pln_dir = os.path.join(current_dir, 'PLN')
if pln_dir not in sys.path:
    sys.path.insert(0, pln_dir)

from recommender import RecommendationEngine


# Configuracion de la pagina
st.set_page_config(
    page_title="Sistema de Recomendacion de Empleos",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommendation_engine():
    # Carga el motor de recomendacion (se ejecuta una sola vez)
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        engine = RecommendationEngine()
        return engine
    finally:
        sys.stdout = old_stdout


def format_score(score: float) -> str:
    # Formatea el score de similitud como porcentaje
    return f"{score * 100:.1f}%"


def get_score_color(score: float) -> str:
    # Retorna un color segun el score
    if score >= 0.7:
        return "#28a745"
    elif score >= 0.5:
        return "#ffc107"
    else:
        return "#dc3545"


def extract_text_from_pdf(pdf_file) -> str:
    # Extrae texto de un archivo PDF
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error al leer el PDF: {str(e)}")
        return ""


def main():
    st.markdown('<p class="main-header">Sistema de Recomendacion de Empleos</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Encuentra las mejores ofertas laborales usando IA</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Configuracion")
        k = st.slider("Numero de recomendaciones", min_value=5, max_value=20, value=10, step=1)
        st.markdown("---")
        st.subheader("Informacion del Sistema")
        
        with st.spinner("Cargando motor..."):
            try:
                engine = load_recommendation_engine()
                stats = engine.get_statistics()
                st.metric("Total de Ofertas", f"{stats['total_jobs']:,}")
                st.metric("Dimension Embeddings", stats['embedding_dimension'])
                st.markdown("**Categorias disponibles:**")
                for source, count in stats['sources'].items():
                    categoria = source.replace('vectors_', '').replace('.pkl', '').title()
                    st.write(f"• {categoria}: {count}")
            except Exception as e:
                st.error(f"Error al cargar el sistema: {str(e)}")
                return
        st.markdown("---")
        st.info("**Tip:** Describe tu perfil con detalle para mejores recomendaciones.")
    
    # Main content
    st.markdown("## Describe tu Perfil Profesional")
    tab1, tab2 = st.tabs(["Escribir Perfil", "Subir CV (PDF)"])
    
    perfil_texto = ""
    
    with tab1:
        perfil_texto = st.text_area(
            "Tu perfil profesional:",
            value="",
            height=200,
            placeholder="Describe tu experiencia, habilidades tecnicas, conocimientos, proyectos realizados y tipo de trabajo que buscas...",
            key="perfil_manual"
        )
    
    with tab2:
        st.markdown("### Sube tu CV en formato PDF")
        st.info("El sistema extraera automaticamente el texto de tu CV.")
        uploaded_file = st.file_uploader("Selecciona tu archivo PDF:", type=['pdf'])
        if uploaded_file is not None:
            with st.spinner("Extrayendo texto del PDF..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    st.success(f"Texto extraido exitosamente ({len(extracted_text)} caracteres)")
                    with st.expander("Ver texto extraido del CV"):
                        st.text_area("Contenido extraido:", value=extracted_text, height=300, disabled=True)
                    perfil_texto = extracted_text
                else:
                    st.error("No se pudo extraer texto del PDF.")
        else:
            st.markdown("**Nota:** No has subido ningun archivo aun.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        buscar = st.button("Buscar Ofertas", use_container_width=True, type="primary")
    
    if buscar:
        if not perfil_texto or perfil_texto.strip() == "":
            st.warning("Por favor, describe tu perfil profesional.")
            return
        
        with st.spinner("Analizando tu perfil y buscando ofertas relevantes..."):
            try:
                ofertas = engine.recomendar(perfil_texto, k=k, verbose=False)
                st.markdown("---")
                st.markdown("## Resultados")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ofertas Encontradas", len(ofertas))
                with col2:
                    avg_score = sum(o['score'] for o in ofertas) / len(ofertas)
                    st.metric("Score Promedio", format_score(avg_score))
                with col3:
                    best_score = ofertas[0]['score'] if ofertas else 0
                    st.metric("Mejor Match", format_score(best_score))
                
                st.markdown("---")
                st.markdown("## Top Ofertas Recomendadas")
                
                for i, oferta in enumerate(ofertas, 1):
                    score_color = get_score_color(oferta['score'])
                    categoria = oferta['_source_file'].replace('vectors_', '').replace('.pkl', '').title()
                    with st.expander(f"**{i}. {oferta['title']}** - Match: {format_score(oferta['score'])}", expanded=(i <= 3)):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Categoria:** {categoria}")
                            st.markdown(f"**ID:** {oferta['id']}")
                        with col2:
                            st.markdown(f'<div style="background-color: {score_color}; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; font-weight: bold;">{format_score(oferta["score"])}</div>', unsafe_allow_html=True)
                        st.markdown("**Descripcion:**")
                        st.write(oferta['description'])
                        st.markdown(f"**Fuente:** {oferta['source']} | **Fecha:** {oferta['scraped_at'][:10]}")
                
                st.success("Busqueda completada exitosamente")
            except Exception as e:
                st.error(f"Error al procesar la busqueda: {str(e)}")
                st.exception(e)
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>Sistema de Recomendacion CBF | Powered by FAISS + Sentence Transformers</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
