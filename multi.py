import streamlit as st
import PyPDF2
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import hashlib
import pickle
import time
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Configurar a chave da API do OpenAI usando st.secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Inicializar o modelo GPT-4
gpt4 = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# O resto do código permanece o mesmo
def get_file_hash(file):
    file.seek(0)
    return hashlib.md5(file.read()).hexdigest()

def extract_text_from_pdf(file, progress_bar, status_text):
    try:
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            text = ""
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    # Se não houver texto, tenta OCR
                    image = page.to_image()
                    page_text = pytesseract.image_to_string(image.original) + "\n"
                text += page_text

                # Atualizar o progresso
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Extraindo texto: {progress:.0%} concluído")
        return text
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
        return None

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

# Template para resumo de cada chunk
chunk_summary_template = """
Analise o seguinte texto de um documento jurídico e forneça um resumo conciso, focando nos pontos essenciais:

{text}

Instruções específicas:
1. Concentre-se no teor principal do documento, extraindo as informações mais relevantes.
2. Descarte dados de assinatura eletrônica.
3. Foque especialmente em manifestações de advogados(as), promotores(as), juízes(as) e outras autoridades.
4. Preste atenção especial aos seguintes elementos, se presentes:
   - Resposta à acusação
   - Alegações finais
   - Manifestações da defesa
   - Transcrição de audiência (se contida nas alegações finais)
5. Identifique e resuma os argumentos principais apresentados no texto.
6. Extraia citações relevantes que apoiem os pontos principais.
7. Se houver decisões ou conclusões, destaque-as.

Resumo:
"""

{text}

Resumo conciso (máximo de 3 frases):
"""

CHUNK_SUMMARY_PROMPT = PromptTemplate(template=chunk_summary_template, input_variables=["text"])

# Template para análise final
final_analysis_template = """
Você é um assistente jurídico especializado em análise de documentos legais. Com base nos resumos fornecidos de um documento jurídico extenso, por favor:

1. Identifique as principais partes envolvidas (juiz, advogados, promotor, réu, etc.).
2. Resuma os pontos principais do caso, incluindo alegações, evidências e decisões.
3. Destaque quaisquer manifestos importantes do juiz, advogados ou promotor.
4. Identifique os principais argumentos legais apresentados.
5. Resuma o resultado ou estado atual do processo.
6. Destaque quaisquer pontos únicos ou incomuns sobre o caso.

Resumos dos chunks:
{text}

Forneça uma análise estruturada e concisa, evitando repetições e focando nas informações mais relevantes:
"""

FINAL_ANALYSIS_PROMPT = PromptTemplate(template=final_analysis_template, input_variables=["text"])

def analyze_document(text, progress_bar, status_text):
    chunks = split_text(text)

    # Criar chain para resumo de chunks
    chunk_chain = load_summarize_chain(
        gpt4,
        chain_type="stuff",
        prompt=CHUNK_SUMMARY_PROMPT
    )

    # Resumir cada chunk
    summaries = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        summary = chunk_chain.run([chunk])
        summaries.append((i+1, summary))  # Adicionando o número do chunk

        # Atualizar o progresso
        progress = (i + 1) / total_chunks
        progress_bar.progress(progress)
        status_text.text(f"Analisando documento: {progress:.0%} concluído")

    # Análise final
    status_text.text("Gerando análise final...")
    final_chain = load_summarize_chain(
        gpt4,
        chain_type="stuff",
        prompt=FINAL_ANALYSIS_PROMPT
    )

    final_summary = final_chain.run([Document(page_content="\n".join([s[1] for s in summaries]))])
    if isinstance(final_summary, list):
        final_summary = " ".join(final_summary)

    return summaries, final_summary

def generate_pdf(summaries, final_summary):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4))  # 4 is for justify alignment

    flowables = []

    # Título
    flowables.append(Paragraph("ANÁLISE DO DOCUMENTO JURÍDICO", styles['Title']))
    flowables.append(Spacer(1, 0.5*inch))

    # Resumo Final
    flowables.append(Paragraph("Resumo Geral", styles['Heading1']))
    if isinstance(final_summary, list):
        for item in final_summary:
            flowables.append(Paragraph(item, styles['Justify']))
            flowables.append(Spacer(1, 0.1*inch))
    else:
        flowables.append(Paragraph(final_summary, styles['Justify']))
    flowables.append(Spacer(1, 0.25*inch))

    # Detalhes dos Chunks
    flowables.append(PageBreak())
    flowables.append(Paragraph("Detalhes da Análise", styles['Heading1']))
    flowables.append(Spacer(1, 0.25*inch))

    for chunk_num, summary in summaries:
        flowables.append(Paragraph(f"Parte {chunk_num} do Documento:", styles['Heading2']))
        flowables.append(Paragraph(summary, styles['Justify']))
        flowables.append(Spacer(1, 0.25*inch))

    # Conclusão
    flowables.append(PageBreak())
    flowables.append(Paragraph("Conclusão", styles['Heading1']))
    flowables.append(Paragraph(
        "Este relatório apresenta uma análise detalhada do documento jurídico, "
        "começando com um resumo geral seguido por análises específicas de cada parte do documento. "
        "A leitura sequencial deste relatório fornece uma compreensão completa e lógica dos fatos e atos do processo.",
        styles['Justify']
    ))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

def chat_with_document(query, summaries, final_summary):
    # Combine todos os resumos e a análise final em um único contexto
    context = final_summary + "\n\n" + "\n\n".join([f"Parte {i}: {summary}" for i, summary in summaries])

    # Template para a resposta do chat
    chat_template = """
    Com base na seguinte análise de um documento jurídico:

    {context}

    Por favor, responda à seguinte pergunta:

    {query}

    Resposta:
    """

    chat_prompt = PromptTemplate(template=chat_template, input_variables=["context", "query"])

    # Usar LLMChain para gerar a resposta
    chat_chain = LLMChain(llm=gpt4, prompt=chat_prompt)

    response = chat_chain.run(context=context, query=query)
    return response

# Configuração da página Streamlit
st.set_page_config(page_title="Analisador Avançado de Documentos Jurídicos", layout="wide")

st.title("Analisador Avançado de Documentos Jurídicos")

uploaded_file = st.file_uploader("Escolha um arquivo PDF", type="pdf")

if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file)
    cache_file = f"cache_{file_hash}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            summaries, final_summary = pickle.load(f)
        st.success("Análise carregada do cache!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        with st.spinner("Analisando o documento..."):
            full_text = extract_text_from_pdf(uploaded_file, progress_bar, status_text)

            if full_text is None:
                st.error("Não foi possível extrair o texto do PDF. Verifique se o arquivo não está corrompido ou protegido.")
            else:
                summaries, final_summary = analyze_document(full_text, progress_bar, status_text)

                with open(cache_file, 'wb') as f:
                    pickle.dump((summaries, final_summary), f)

        progress_bar.empty()
        status_text.empty()
        st.success("Análise concluída!")

    st.subheader("Análise do Documento")
    st.write(final_summary)

    st.info("O relatório completo, incluindo resumos detalhados, está disponível no arquivo PDF.")

    pdf_buffer = generate_pdf(summaries, final_summary)
    st.download_button(
        label="Baixar Relatório Completo em PDF",
        data=pdf_buffer,
        file_name="analise_documento_juridico.pdf",
        mime="application/pdf"
    )

    with st.expander("Ver Resumos Detalhados"):
        for i, summary in summaries:
            st.write(f"Parte {i} do Documento: {summary}")

    # Adicionar funcionalidade de chat
    st.subheader("Chat com o Documento")
    user_query = st.text_input("Faça uma pergunta sobre o documento:")
    if st.button("Enviar Pergunta"):
        if user_query:
            with st.spinner("Processando sua pergunta..."):
                response = chat_with_document(user_query, summaries, final_summary)
            st.write("Resposta:", response)
        else:
            st.warning("Por favor, insira uma pergunta antes de enviar.")

# Sidebar
st.sidebar.title("Sobre a Ferramenta")
st.sidebar.info(
    "Esta ferramenta utiliza inteligência artificial avançada para analisar documentos jurídicos extensos, "
    "fornecendo resumos concisos e insights relevantes. Ela é especialmente útil para documentos com mais de 200 páginas, "
    "permitindo uma rápida compreensão dos pontos principais sem a necessidade de ler todo o documento."
)
st.sidebar.title("Desenvolvido por")
st.sidebar.write("Nome: João Valério")
st.sidebar.write("Cargo: Juiz de Direito")

# Adicionar informações sobre o uso da ferramenta
st.sidebar.title("Como Usar")
st.sidebar.write(
    "1. Faça o upload de um documento PDF.\n"
    "2. Aguarde a análise ser concluída.\n"
    "3. Leia o resumo e insights gerados.\n"
    "4. Baixe o relatório completo em PDF.\n"
    "5. Explore os resumos detalhados se necessário.\n"
    "6. Use o chat para fazer perguntas específicas sobre o documento."
)

# Adicionar aviso sobre o tempo de processamento
st.sidebar.warning(
    "O tempo de processamento pode variar dependendo do tamanho e complexidade do documento. "
    "Documentos muito extensos podem levar vários minutos para serem analisados."
)

# Adicionar informações de contato ou suporte
st.sidebar.title("Suporte")
st.sidebar.info(
    "Para dúvidas ou suporte, entre em contato:\n"
    "Email: joao.moura@tjpa.jus.br"
)
