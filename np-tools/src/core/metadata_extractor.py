import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document


class MetadataExtractor:
    """
    Extrae metadatos en 3 bloques (adjudicación, solvencia, condiciones especiales)
    usando SOLO Ollama (LLM + embeddings). El host se toma de OLLAMA_HOST o default.
    """

    def __init__(
        self,
        ollama_llm: str = "llama3.1",
        ollama_embed_model: str = "mxbai-embed-large",
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        retriever_k: int = 10,
    ):
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_llm = ollama_llm
        self.ollama_embed_model = ollama_embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k

        # LLM y embeddings de Ollama (ambos apuntan al mismo host/base_url)
        # Optimizado: num_predict reducido para respuestas más rápidas
        self.model = Ollama(
            model=self.ollama_llm,
            base_url=self.ollama_host,
            temperature=0.3,
            num_predict=2000,  # Reducido de 8000 a 2000 para mayor velocidad
        )
        self.embeddings = OllamaEmbeddings(
            model=self.ollama_embed_model,
            base_url=self.ollama_host,
        )

        self.prompt = self._build_prompt_template()

    # ----------------------- Prompt -----------------------
    def _build_prompt_template(self) -> PromptTemplate:
        template = """Eres un experto en derecho administrativo. Extrae COMPLETAMENTE la información solicitada del contexto.

REGLAS:
- Copia el texto LITERALMENTE del documento, sin resumir
- Incluye TODO: listas, fórmulas, cifras, porcentajes, subsecciones
- Si no hay información relevante, responde: [SIN INFORMACION]

Contexto:
{context}

Pregunta:
{question}

Respuesta (copia literal del texto encontrado):"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    # ----------------------- Helpers -----------------------
    @staticmethod
    def _format_content(text: str) -> str:
        """Si el texto es XML, lo convierte a texto plano; si no, lo devuelve tal cual."""
        try:
            t = (text or "").strip()
            if t.startswith("<"):
                c_et = ET.fromstring(t)
                return ET.tostring(c_et, method="text", encoding="utf-8").decode("utf-8")
            return t
        except Exception:
            return text or ""

    def _create_documents(self, text: str) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_text(text)
        # metadatos simples para depuración opcional
        return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

    @staticmethod
    def _clean_result(text: str) -> str:
        """
        Limpieza de texto mejorada: elimina markdown, deduplica y elimina "no existe".
        """
        if not text:
            return ""
        
        # Eliminar caracteres markdown
        text = re.sub(r'\*\*', '', text)  # Eliminar **
        text = re.sub(r'\*', '', text)    # Eliminar *
        text = re.sub(r'###', '', text)    # Eliminar ###
        text = re.sub(r'##', '', text)     # Eliminar ##
        text = re.sub(r'#', '', text)      # Eliminar #
        text = re.sub(r'__', '', text)     # Eliminar __
        text = re.sub(r'_', '', text)      # Eliminar _
        
        # Eliminar espacios múltiples excesivos
        text = re.sub(r'\s{3,}', ' ', text)
        text = text.strip()
        
        # DEDUPLICACIÓN: Eliminar párrafos y sentencias duplicadas
        # Dividir por oraciones (puntos, exclamaciones, interrogaciones)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Eliminar duplicados manteniendo el orden
        seen = set()
        unique_sentences = []
        for sent in sentences:
            sent_clean = sent.strip()
            # Normalizar para comparar (lowercase, sin espacios extra)
            sent_key = re.sub(r'\s+', ' ', sent_clean).lower()[:100]  # Primeros 100 chars para comparar
            
            if sent_key not in seen and len(sent_clean) > 10:  # Ignorar frases muy cortas
                seen.add(sent_key)
                unique_sentences.append(sent_clean)
        
        text = ' '.join(unique_sentences)
        
        # Detectar [SIN INFORMACION] y devolver string vacío
        if '[SIN INFORMACION]' in text.upper():
            return ""
        
        # Si la respuesta indica que no existe información, devolver string vacío
        no_info_patterns = [
            r'no.*existe.*criterio.*documento',
            r'no.*se.*establece',
            r'no.*se.*mencionan',
            r'no.*se.*exigen',
            r'no.*hay.*información',
            r'absolutamente.*no.*encuentro',
            r'ninguna.*información.*relevante',
            r'no.*aparece.*documento',
            r'no.*contiene.*información',
            r'no.*se.*han.*establecido',
            r'no.*se.*ha.*establecido'
        ]
        
        # Comprobar si la respuesta es muy corta y contiene indicadores de "no existe"
        if len(text) < 200:  # Si es muy corta
            for pattern in no_info_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return ""  # Devolver string vacío
        
        return text

    # ----------------------- Pipeline principal -----------------------
    def extract_metadata_from_text(self, text: str) -> Dict[str, str]:
        """
        Entrada: texto largo (string). Salida: dict con 3 claves:
        - criterios_adjudicacion
        - criterios_solvencia
        - condiciones_especiales
        
        Usa RetrievalQA con 3 consultas separadas para extraer cada categoría.
        """
        formatted = self._format_content(text)
        if not formatted:
            return {"error": "Texto vacío o inválido."}

        docs = self._create_documents(formatted)
        if not docs:
            return {"error": "El contenido quedó vacío después de procesar."}

        # Vector store con embeddings de Ollama
        try:
            vector_storage = FAISS.from_documents(docs, self.embeddings)
        except Exception as e:
            return {
                "error": (
                    f"No se pudo inicializar FAISS con embeddings de Ollama en {self.ollama_host}. "
                    f"¿Está el servidor activo? Detalle: {str(e)}"
                )
            }

        retriever = vector_storage.as_retriever(
            search_kwargs={"k": self.retriever_k})

        # Crear cadena QA con load_qa_chain
        qa_chain = load_qa_chain(
            llm=self.model,
            chain_type="stuff",
            prompt=self.prompt
        )

        # Consultas específicas para cada categoría (optimizadas para ser más directas)
        queries = {
            "criterios_adjudicacion": (
                "Extrae los criterios de adjudicación o evaluación de ofertas: "
                "subcriterios, puntuaciones, baremos, fórmulas, ponderación, reglas de desempate."
            ),
            "criterios_solvencia": (
                "Extrae los criterios de solvencia, capacidad económica, técnica o profesional: "
                "requisitos económicos, técnicos, medios materiales, humanos, experiencia, clasificación."
            ),
            "condiciones_especiales": (
                "Extrae las condiciones especiales de ejecución: "
                "criterios sociales, medioambientales, sostenibilidad, responsabilidad social."
            )
        }

        resultado = {}
        
        # Procesar cada consulta por separado
        for key, query in queries.items():
            try:
                # Obtener documentos relevantes del retriever
                relevant_docs = retriever.get_relevant_documents(query)
                
                # Ejecutar la cadena QA con los documentos y la pregunta
                result = qa_chain.invoke({
                    "input_documents": relevant_docs,
                    "question": query
                })
                
                # Extraer el texto del resultado
                result_text = result.get("output_text", "") if isinstance(result, dict) else str(result)
                resultado[key] = self._clean_result(result_text)
            except Exception as e:
                resultado[key] = ""
                # Log del error pero continuar con las demás consultas
                print(f"Error procesando {key}: {str(e)}")

        return resultado
