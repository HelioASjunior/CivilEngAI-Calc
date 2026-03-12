import io
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from PIL import Image, ImageOps
from streamlit_paste_button import paste_image_button
from PyPDF2 import PdfReader


# Funcoes matematicas puras de engenharia civil (sem efeito colateral).
# ---------- Civil Engineering pure functions ----------
def calc_beam_required_steel_area_m2(
    m_sd_knm: float,
    d_mm: float,
    f_yd_mpa: float,
    z_factor: float = 0.9,
) -> float:
    """Simplified required tensile steel area for RC beam.

    Uses A_s = Msd / (z * fyd), with z ~= 0.9*d.
    Returns area in m^2.
    """
    # Converte momento de kN.m para N.m.
    m_sd_nm = m_sd_knm * 1_000.0
    # Converte altura util de mm para m.
    d_m = d_mm / 1000.0
    # Calcula o braco de alavanca aproximado.
    z_m = z_factor * d_m
    # Converte tensao de MPa para Pa.
    f_yd_pa = f_yd_mpa * 1_000_000.0
    # Retorna area de aco necessaria em m2.
    return m_sd_nm / (z_m * f_yd_pa)


def calc_void_index(gs: float, gamma_d_kn_m3: float, gamma_w_kn_m3: float = 9.81) -> float:
    """Void ratio index e = (Gs * gamma_w / gamma_d) - 1."""
    # Aplica formula direta do indice de vazios.
    return (gs * gamma_w_kn_m3 / gamma_d_kn_m3) - 1.0


def calc_manning_flow_q(
    area_m2: float,
    hydraulic_radius_m: float,
    slope_m_m: float,
    roughness_n: float,
) -> float:
    """Open channel flow by Manning equation: Q = (1/n) A R^(2/3) S^(1/2)."""
    # Calcula vazao em canal aberto pela equacao de Manning.
    return (1.0 / roughness_n) * area_m2 * (hydraulic_radius_m ** (2.0 / 3.0)) * (slope_m_m ** 0.5)


# Estrutura para guardar imagem pronta para enviar para API.
# ---------- Multimodal payload helpers ----------
@dataclass
class PreparedImage:
    # Bytes comprimidos da imagem.
    bytes_data: bytes
    # Tipo MIME do arquivo.
    mime_type: str
    # Objeto PIL para preview e envio.
    pil_image: Image.Image


def preprocess_image(uploaded_file) -> Optional[PreparedImage]:
    # Se nao houver arquivo, nao processa nada.
    if uploaded_file is None:
        return None

    # Abre, corrige orientacao EXIF e padroniza em RGB.
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Keep detail while controlling payload size for API calls.
    max_side = 1600
    image.thumbnail((max_side, max_side))

    # Converte para JPEG comprimido para reduzir payload.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)

    # Retorna pacote com bytes, mime e imagem pronta.
    return PreparedImage(bytes_data=buffer.getvalue(), mime_type="image/jpeg", pil_image=image)


def preprocess_pil_image(image: Image.Image) -> PreparedImage:
    # Normaliza imagem vinda da area de transferencia.
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized.thumbnail((1600, 1600))

    # Salva em JPEG para envio mais eficiente.
    buffer = io.BytesIO()
    normalized.save(buffer, format="JPEG", quality=90, optimize=True)
    # Retorna estrutura padronizada.
    return PreparedImage(bytes_data=buffer.getvalue(), mime_type="image/jpeg", pil_image=normalized)


def build_engineering_system_instruction(pdf_context: Optional[str] = None) -> str:
    # Instrucao de sistema fixa para orientar o modelo em respostas tecnicas.
    base_instruction = (
        "Voce e um Engenheiro Civil Senior Multidisciplinar. "
        "Escopo total: responda sobre todas as areas da Engenharia Civil, incluindo Estruturas "
        "(Concreto, Aco, Madeira), Hidraulica e Saneamento, Geotecnia (Solos e Fundacoes), "
        "Estradas e Transportes, Gerenciamento de Obras e Materiais de Construcao. "
        "Normas tecnicas: use sempre a norma ABNT pertinente ao assunto. "
        "Exemplos: para perda de carga, aplique Hidraulica (Bernoulli e Darcy-Weisbach quando pertinente); "
        "para vigas e elementos estruturais, aplique NBR 6118, NBR 8800 e demais normas aplicaveis. "
        "Rigor matematico: mantenha precisao absoluta e apresente o passo a passo das formulas antes do resultado final. "
        "Unidades: use estritamente o SI no resultado final, aceitando entradas em outras unidades "
        "(como tf e kgf) com conversao explicita e correta. "
        "Postura: seja direto, tecnico e consultivo. "
        "Se houver imagem (planta, diagrama de momentos, tabela, print tecnico), analise os dados visuais "
        "com foco em seguranca estrutural, eficiencia tecnica e conformidade normativa. "
        "Formato obrigatorio da resposta em Markdown: "
        "(1) resumo do problema, (2) premissas adotadas, (3) tabela de dados de entrada com unidades, "
        "(4) formulas em LaTeX, (5) calculo passo a passo com conversoes de unidades, "
        "(6) verificacoes normativas ABNT, (7) conclusao objetiva com recomendacoes tecnicas."
    )
    
    if pdf_context:
        pdf_instruction = (
            "\n\n[PRIORIDADE MAXIMA - HIERARQUIA DE BUSCA COM TAGS OBRIGATORIAS]\n"
            "Para CADA pergunta, siga rigorosamente esta ordem:\n"
            "1. ANALISE PRIMEIRO o conteudo do MANUAL TECNICO a seguir.\n"
            "2. Se encontrar a resposta NO MANUAL:\n"
            "   - INICIE A RESPOSTA COM [FONTE: PDF]\n"
            "   - Incluia uma citacao DIRETA do texto entre aspas duplas.\n"
            "   - Exemplo: [FONTE: PDF] De acordo com o manual tecnico: \"texto direto extraido do PDF\".\n"
            "3. Se a informacao NAO constar no manual ou for insuficiente:\n"
            "   - INICIE A RESPOSTA COM [FONTE: GLOBAL]\n"
            "   - Inclua aviso: \"Esta informacao nao consta no manual especifico, mas seguindo as normas gerais de engenharia...\"\n"
            "4. Em CONFLITO DE INFORMACAO: PRIORIZE O QUE ESTA NO MANUAL (use [FONTE: PDF]).\n"
            "5. Apos a resposta principal, adicione uma secao [CONTEXTO DA CONSULTA]:\n"
            "   - Se [FONTE: PDF]: liste o numero estimado de paragrafos/secoes consultados do PDF.\n"
            "   - Se [FONTE: GLOBAL]: mencione quais normas ABNT ou conceitos foram aplicados.\n\n"
            "[MANUAL TECNICO CARREGADO]:\n"
            f"{pdf_context[:8000]}"  # Limita a 8000 caracteres para não sobrecarregar
        )
        return base_instruction + pdf_instruction
    else:
        return base_instruction


def list_available_gemini_models(client: genai.Client) -> list[str]:
    # Monta lista com nomes de modelos disponiveis para a chave.
    model_names = []
    for model in client.models.list():
        name = getattr(model, "name", "")
        # Ignora itens sem nome valido.
        if not name:
            continue
        # Remove prefixo do recurso e guarda so o nome final.
        model_names.append(name.split("/")[-1])
    return model_names


def diagnose_and_pick_model(api_key: str) -> tuple[list[str], Optional[str], Optional[str]]:
    # Cria cliente Gemini com a chave informada.
    client = genai.Client(api_key=api_key)

    try:
        # Tenta listar modelos habilitados na conta.
        available_models = list_available_gemini_models(client)
    except Exception as exc:
        # Retorna erro amigavel caso falhe a listagem.
        return [], None, f"Falha ao listar modelos da API: {exc}"

    # Se nao vier nenhum modelo, orienta criar nova chave/projeto.
    if not available_models:
        return [], None, "Sua chave não tem modelos ativos. Crie uma nova chave em um NOVO PROJETO no AI Studio."

    # Prioriza explicitamente gemini-1.5-flash; se nao existir, usa outro Flash.
    preferred_models = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
    ]
    selected_model = None
    for preferred in preferred_models:
        match = next((m for m in available_models if m.lower() == preferred), None)
        if match:
            selected_model = match
            break

    if not selected_model:
        flash_candidates = [m for m in available_models if "flash" in m.lower()]
        selected_model = flash_candidates[0] if flash_candidates else available_models[0]
    return available_models, selected_model, None


def is_not_found_error(exc: Exception) -> bool:
    # Verifica se o erro e de recurso/modelo nao encontrado.
    message = str(exc).upper()
    return "404" in message or "NOT_FOUND" in message


def extract_text_from_pdf(pdf_file) -> str:
    """Extrai texto de um arquivo PDF enviado pelo usuario."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as exc:
        return f"Erro ao extrair PDF: {exc}"


def ask_gemini(api_key: str, model_name: str, prompt: str, prepared_image: Optional[PreparedImage], pdf_context: Optional[str] = None) -> str:
    # Cria cliente Gemini.
    client = genai.Client(api_key=api_key)

    # Conteudo base enviado ao modelo (texto do usuario).
    contents = [prompt]
    # Se houver imagem, envia junto com o texto.
    if prepared_image:
        contents.append(prepared_image.pil_image)

    # Gera resposta com instrucao de sistema tecnica (com ou sem contexto PDF).
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config={"system_instruction": build_engineering_system_instruction(pdf_context)},
    )
    # Retorna texto da resposta ou fallback padrao.
    return response.text if getattr(response, "text", None) else "Sem resposta da API."


def parse_assistant_response(response_text: str) -> dict:
    """Parseia resposta da IA para extrair origem, citacoes e contexto.
    
    Retorna dict com:
    - 'source': 'PDF' ou 'GLOBAL'
    - 'citation': trecho entre aspas (se houver)
    - 'context': detalhes da consulta
    - 'main_text': resposta principal (limpa das tags)
    """
    result = {
        "source": "GLOBAL",
        "citation": "",
        "context": "",
        "main_text": response_text
    }
    
    # Detecta tags de origem.
    if "[FONTE: PDF]" in response_text:
        result["source"] = "PDF"
        response_text = response_text.replace("[FONTE: PDF]", "").strip()
    elif "[FONTE: GLOBAL]" in response_text:
        result["source"] = "GLOBAL"
        response_text = response_text.replace("[FONTE: GLOBAL]", "").strip()
    
    # Extrai trecho citado (entre aspas duplas).
    import re
    citations = re.findall(r'"([^"]+)"', response_text)
    if citations:
        result["citation"] = citations[0]  # Primeira citacao encontrada.
    
    # Extrai contexto da consulta (secao apos [CONTEXTO DA CONSULTA]).
    if "[CONTEXTO DA CONSULTA]" in response_text:
        parts = response_text.split("[CONTEXTO DA CONSULTA]")
        result["main_text"] = parts[0].strip()
        result["context"] = parts[1].strip() if len(parts) > 1 else ""
    else:
        result["main_text"] = response_text
    
    return result


def is_quota_error(exc: Exception) -> bool:
    # Verifica se o erro indica limite de uso/quota esgotada.
    message = str(exc).upper()
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "RESOURCE EXHAUSTED" in message


def add_report_to_history(title: str, summary: str) -> None:
    # Registra um item no historico de relatorios para acesso rapido na sidebar.
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    history = st.session_state.report_history
    history.insert(0, {"title": title, "summary": summary, "timestamp": timestamp})
    # Mantem apenas os ultimos 8 registros.
    st.session_state.report_history = history[:8]


def generate_chat_title(user_prompt: str) -> str:
    # Gera titulo curto (max 5 palavras) com base na primeira frase do usuario.
    first_sentence = user_prompt.strip().split("\n")[0]
    for separator in [".", "?", "!", ";", ":"]:
        first_sentence = first_sentence.split(separator)[0]

    words = [w for w in first_sentence.replace(",", " ").split() if w]
    if not words:
        return "Nova Conversa"

    short_title = " ".join(words[:5]).strip()
    return short_title if short_title else "Nova Conversa"


def ensure_active_conversation() -> None:
    # Garante que sempre exista uma conversa ativa para o chat principal.
    if not st.session_state.historico:
        st.session_state.historico = [{"titulo": "Nova Conversa", "mensagens": []}]
        st.session_state.conversa_ativa = 0

    if st.session_state.conversa_ativa is None:
        st.session_state.conversa_ativa = 0

    if st.session_state.conversa_ativa >= len(st.session_state.historico):
        st.session_state.conversa_ativa = len(st.session_state.historico) - 1


def sync_messages_to_active_conversation() -> None:
    # Sincroniza mensagens da tela com a conversa ativa no historico.
    ensure_active_conversation()
    idx = st.session_state.conversa_ativa
    st.session_state.historico[idx]["mensagens"] = list(st.session_state.messages)


def load_conversation(index: int) -> None:
    # Carrega uma conversa do historico para o chat principal.
    if 0 <= index < len(st.session_state.historico):
        st.session_state.conversa_ativa = index
        st.session_state.messages = list(st.session_state.historico[index]["mensagens"])


def start_new_conversation() -> None:
    # Cria nova conversa vazia sem apagar as anteriores.
    sync_messages_to_active_conversation()
    st.session_state.historico.insert(0, {"titulo": "Nova Conversa", "mensagens": []})
    st.session_state.conversa_ativa = 0
    st.session_state.messages = []
    st.session_state.last_assistant_answer = ""
    st.session_state.clipboard_image = None
    st.session_state.uploader_nonce += 1


def delete_conversation(index: int) -> None:
    # Exclui uma conversa especifica e reajusta conversa ativa.
    if not (0 <= index < len(st.session_state.historico)):
        return

    active_idx = st.session_state.conversa_ativa
    st.session_state.historico.pop(index)

    if not st.session_state.historico:
        st.session_state.historico = [{"titulo": "Nova Conversa", "mensagens": []}]
        st.session_state.conversa_ativa = 0
        st.session_state.messages = []
        return

    if active_idx == index:
        st.session_state.conversa_ativa = 0
        st.session_state.messages = list(st.session_state.historico[0]["mensagens"])
    elif active_idx > index:
        st.session_state.conversa_ativa = active_idx - 1


def clear_all_conversations() -> None:
    # Remove todo o historico e inicia uma conversa nova.
    st.session_state.historico = [{"titulo": "Nova Conversa", "mensagens": []}]
    st.session_state.conversa_ativa = 0
    st.session_state.messages = []
    st.session_state.last_assistant_answer = ""
    st.session_state.confirm_clear_conversations_flag = False


def cb_set_current_page(page: str) -> None:
    st.session_state.current_page = page


def cb_start_new_chat() -> None:
    start_new_conversation()
    st.session_state.current_page = "💬 Assistente Técnico (Chat)"


def cb_open_chat(index: int) -> None:
    sync_messages_to_active_conversation()
    load_conversation(index)
    st.session_state.current_page = "💬 Assistente Técnico (Chat)"


def cb_delete_chat(index: int) -> None:
    delete_conversation(index)
    st.session_state.current_page = "💬 Assistente Técnico (Chat)"


def cb_request_clear_all_chats() -> None:
    st.session_state.confirm_clear_conversations_flag = True
    st.session_state.current_page = "💬 Assistente Técnico (Chat)"


def cb_confirm_clear_all_chats() -> None:
    clear_all_conversations()
    st.session_state.current_page = "💬 Assistente Técnico (Chat)"


def cb_cancel_clear_all_chats() -> None:
    st.session_state.confirm_clear_conversations_flag = False
    st.session_state.current_page = "💬 Assistente Técnico (Chat)"


def cb_clear_active_chat_messages() -> None:
    st.session_state.messages = []
    st.session_state.last_assistant_answer = ""
    sync_messages_to_active_conversation()


# Funcoes de interface (Streamlit).
# ---------- UI ----------
def render_theme(theme_mode: str = "light", full_chat: bool = False, sidebar_collapsed: bool = False) -> None:
    # Ajustes de layout para modo normal ou chat em tela mais ampla.
    top_padding = "0.5rem" if full_chat else "1.0rem"
    max_width = "1420px" if full_chat else "1240px"
    full_chat_css = "[data-testid=\"stVerticalBlock\"]:has([data-testid=\"stChatMessage\"]) {padding-top: 0.15rem;}" if full_chat else ""
    collapsed_sidebar_css = """
    [data-testid="stSidebar"] {margin-left: -22rem;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    """ if (full_chat and sidebar_collapsed) else ""

    # Paleta principal: azul escuro profundo, cinza ardosia e acentos de resultado.
    is_dark = theme_mode == "dark"
    if is_dark:
        app_bg = "radial-gradient(circle at 8% 0%, rgba(59,130,246,0.15), transparent 35%), linear-gradient(160deg, #0B1220 0%, #1E293B 52%, #0F172A 100%)"
        base_text = "#E2E8F0"
        muted_text = "#94A3B8"
        sidebar_bg = "linear-gradient(180deg, #0F172A 0%, #1E293B 100%)"
        card_bg = "rgba(30, 41, 59, 0.58)"
        card_border = "rgba(148, 163, 184, 0.30)"
        tab_bg = "rgba(51, 65, 85, 0.60)"
        tab_border = "rgba(148, 163, 184, 0.35)"
        tab_text = "#E2E8F0"
        tab_active_bg = "#2563EB"
        input_bg = "rgba(15, 23, 42, 0.75)"
        input_text = "#F8FAFC"
        input_border = "rgba(148, 163, 184, 0.45)"
        assistant_chat_bg = "rgba(15, 23, 42, 0.82)"
        user_chat_bg = "rgba(37, 99, 235, 0.30)"
        metric_value = "#22C55E"
        button_bg = "linear-gradient(90deg, #1D4ED8 0%, #2563EB 100%)"
        button_hover_bg = "linear-gradient(90deg, #1E40AF 0%, #1D4ED8 100%)"
    else:
        app_bg = "radial-gradient(circle at 8% 0%, rgba(37,99,235,0.12), transparent 35%), linear-gradient(160deg, #F8FAFC 0%, #EEF2F7 52%, #E2E8F0 100%)"
        base_text = "#0F172A"
        muted_text = "#475569"
        sidebar_bg = "linear-gradient(180deg, #1E293B 0%, #0F172A 100%)"
        card_bg = "rgba(255, 255, 255, 0.72)"
        card_border = "rgba(148, 163, 184, 0.42)"
        tab_bg = "rgba(226, 232, 240, 0.85)"
        tab_border = "rgba(100, 116, 139, 0.35)"
        tab_text = "#0F172A"
        tab_active_bg = "#1D4ED8"
        input_bg = "rgba(255, 255, 255, 0.95)"
        input_text = "#0F172A"
        input_border = "rgba(100, 116, 139, 0.45)"
        assistant_chat_bg = "rgba(248, 250, 252, 0.95)"
        user_chat_bg = "rgba(191, 219, 254, 0.75)"
        metric_value = "#059669"
        button_bg = "linear-gradient(90deg, #1D4ED8 0%, #2563EB 100%)"
        button_hover_bg = "linear-gradient(90deg, #1E40AF 0%, #1D4ED8 100%)"

    # CSS global de estilo SaaS profissional.
    css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            :root {
                --eng-base-text: __BASE_TEXT__;
                --eng-muted-text: __MUTED_TEXT__;
                --eng-card-bg: __CARD_BG__;
                --eng-card-border: __CARD_BORDER__;
                --eng-success: __METRIC_VALUE__;
            }

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif !important;
            }

            #MainMenu,
            header[data-testid="stHeader"],
            footer,
            [data-testid="stToolbar"] {
                visibility: hidden;
                height: 0;
                position: fixed;
            }

            .stApp {
                background: __APP_BG__;
                color: var(--eng-base-text);
                transition: background 0.35s ease, color 0.35s ease;
            }

            [data-testid="stSidebar"] {
                background: __SIDEBAR_BG__;
                border-right: 1px solid var(--eng-card-border);
                backdrop-filter: blur(10px);
                transition: background 0.35s ease;
                min-width: 360px;
                max-width: 360px;
            }

            [data-testid="stSidebar"] * {
                color: #F8FAFC !important;
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                padding: 0.35rem 0.55rem 0.8rem 0.55rem;
            }

            .sb-section-title {
                font-size: 0.74rem;
                letter-spacing: 0.09em;
                font-weight: 700;
                color: #CBD5E1 !important;
                opacity: 0.96;
                margin-bottom: 0.25rem;
            }

            .sb-status {
                background: rgba(15, 23, 42, 0.38);
                border: 1px solid rgba(148, 163, 184, 0.34);
                border-radius: 12px;
                padding: 0.58rem 0.7rem;
                margin-top: 0.15rem;
            }

            .sb-status-dot {
                display: inline-block;
                width: 9px;
                height: 9px;
                border-radius: 999px;
                margin-right: 8px;
                background: #22C55E;
                box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.15);
            }

            [data-testid="stSidebar"] [role="radiogroup"] {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.4rem;
            }

            [data-testid="stSidebar"] [role="radiogroup"] > label {
                border: 1px solid rgba(148, 163, 184, 0.45);
                border-radius: 11px;
                padding: 0.36rem 0.52rem;
                background: rgba(15, 23, 42, 0.32);
                justify-content: center;
                min-height: 40px;
                transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
            }

            [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
                border-color: rgba(96, 165, 250, 0.95);
                background: rgba(30, 64, 175, 0.26);
                transform: translateY(-1px);
            }

            [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
                border-color: rgba(96, 165, 250, 0.98);
                background: rgba(37, 99, 235, 0.38);
                box-shadow: 0 8px 16px rgba(30, 64, 175, 0.22);
            }

            .block-container {
                padding-top: __TOP_PADDING__;
                max-width: __MAX_WIDTH__;
            }

            .eng-card,
            [data-testid="stMetric"],
            [data-testid="stFileUploaderDropzone"],
            [data-testid="stChatInput"] {
                background: var(--eng-card-bg);
                border: 1px solid var(--eng-card-border);
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
                backdrop-filter: blur(12px);
            }

            .eng-card {
                padding: 1rem 1.2rem;
                transition: background 0.35s ease, border-color 0.35s ease;
            }

            [data-testid="stMetric"] {
                padding: 0.8rem 1rem;
            }

            [data-testid="stMetricLabel"] {
                color: var(--eng-muted-text) !important;
                font-size: 0.82rem !important;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }

            [data-testid="stMetricValue"] {
                color: var(--eng-success) !important;
                font-size: 1.85rem !important;
                font-weight: 700 !important;
                line-height: 1.1;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                background: __TAB_BG__;
                border-radius: 10px;
                border: 1px solid __TAB_BORDER__;
                color: __TAB_TEXT__;
                padding: 8px 14px;
            }

            .stTabs [aria-selected="true"] {
                background: __TAB_ACTIVE_BG__ !important;
                color: #FFFFFF !important;
                border-color: __TAB_ACTIVE_BG__ !important;
            }

            .stTextInput input,
            .stNumberInput input,
            .stTextArea textarea,
            [data-baseweb="select"] > div,
            [data-testid="stFileUploaderDropzone"],
            [data-testid="stChatInput"] textarea {
                background: __INPUT_BG__ !important;
                color: __INPUT_TEXT__ !important;
                border: 1px solid __INPUT_BORDER__ !important;
                transition: background 0.25s ease, color 0.25s ease, border-color 0.25s ease;
            }

            .stButton > button {
                width: 100%;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.18);
                background: __BUTTON_BG__;
                color: #FFFFFF;
                font-weight: 600;
                transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
                box-shadow: 0 8px 20px rgba(29, 78, 216, 0.25);
            }

            [data-testid="stSidebar"] .stButton > button {
                min-height: 42px;
                border-radius: 13px;
                text-align: left;
                padding-left: 0.85rem;
            }

            /* Botao de acao compacta (ex.: lixeira) com icone centralizado */
            [data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
                min-height: 42px;
                border-radius: 13px;
                padding-left: 0 !important;
                padding-right: 0 !important;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }

            .stButton > button:hover {
                background: __BUTTON_HOVER_BG__;
                transform: translateY(-1px);
                box-shadow: 0 12px 26px rgba(29, 78, 216, 0.35);
            }

            .stButton > button:focus {
                border-color: #60A5FA;
                box-shadow: 0 0 0 0.2rem rgba(96, 165, 250, 0.28);
            }

            [data-testid="stFileUploaderDropzone"] {
                border-width: 1px !important;
                border-style: solid !important;
                border-radius: 15px !important;
                padding: 0.65rem 0.8rem !important;
            }

            [data-testid="stFileUploaderDropzoneInstructions"] {
                font-size: 0.82rem !important;
                opacity: 0.9;
            }

            [data-testid="stChatMessage"] {
                border-radius: 14px;
                border: 1px solid var(--eng-card-border);
                padding: 0.3rem 0.75rem;
                margin-bottom: 0.45rem;
                backdrop-filter: blur(10px);
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                background: __USER_CHAT_BG__;
                border-color: rgba(37, 99, 235, 0.45);
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
                background: __ASSISTANT_CHAT_BG__;
                border-color: var(--eng-card-border);
            }

            .katex,
            .katex * {
                color: var(--eng-base-text) !important;
            }

            h1, h2, h3, h4, h5, h6, p, label, span, li, div {
                color: var(--eng-base-text);
            }

            .stDataFrame, .stTable, .stDataFrame *, .stTable * {
                color: var(--eng-base-text) !important;
                background-color: transparent !important;
            }

            table, th, td {
                color: var(--eng-base-text) !important;
                border-color: var(--eng-card-border) !important;
            }

            __FULL_CHAT_CSS__
            __COLLAPSED_SIDEBAR_CSS__
        </style>
    """
    # Injeta os valores finais no CSS.
    css = (
        css.replace("__TOP_PADDING__", top_padding)
        .replace("__MAX_WIDTH__", max_width)
        .replace("__APP_BG__", app_bg)
        .replace("__BASE_TEXT__", base_text)
        .replace("__SIDEBAR_BG__", sidebar_bg)
        .replace("__CARD_BG__", card_bg)
        .replace("__CARD_BORDER__", card_border)
        .replace("__TAB_BG__", tab_bg)
        .replace("__TAB_BORDER__", tab_border)
        .replace("__TAB_TEXT__", tab_text)
        .replace("__TAB_ACTIVE_BG__", tab_active_bg)
        .replace("__INPUT_BG__", input_bg)
        .replace("__INPUT_TEXT__", input_text)
        .replace("__INPUT_BORDER__", input_border)
        .replace("__MUTED_TEXT__", muted_text)
        .replace("__ASSISTANT_CHAT_BG__", assistant_chat_bg)
        .replace("__USER_CHAT_BG__", user_chat_bg)
        .replace("__METRIC_VALUE__", metric_value)
        .replace("__BUTTON_BG__", button_bg)
        .replace("__BUTTON_HOVER_BG__", button_hover_bg)
        .replace("__FULL_CHAT_CSS__", full_chat_css)
        .replace("__COLLAPSED_SIDEBAR_CSS__", collapsed_sidebar_css)
    )

    # Renderiza CSS no app.
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def render_engineering_header() -> None:
    # Cabeçalho principal da página de cálculos.
    st.markdown("""
    <div class="eng-card">
        <h2 style="margin-bottom:0.3rem;">Plataforma Profissional de Engenharia Civil</h2>
        <p style="margin-top:0;">Entrada técnica, processamento matemático e suporte inteligente multimodal para decisões de projeto.</p>
    </div>
    """, unsafe_allow_html=True)


def render_floating_scientific_calculator(theme_mode: str = "light") -> None:
    # Injeta uma calculadora flutuante no documento pai para nao afetar o layout do Streamlit.
    if theme_mode == "dark":
        fab_bg = "linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%)"
        panel_bg = "rgba(15, 23, 42, 0.55)"
        panel_border = "rgba(148, 163, 184, 0.45)"
        text_color = "#E2E8F0"
        input_bg = "rgba(15, 23, 42, 0.70)"
        button_bg = "rgba(30, 41, 59, 0.55)"
        button_hover = "rgba(59, 130, 246, 0.35)"
        shadow = "0 18px 36px rgba(2, 6, 23, 0.45)"
    else:
        fab_bg = "linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%)"
        panel_bg = "rgba(248, 250, 252, 0.58)"
        panel_border = "rgba(100, 116, 139, 0.35)"
        text_color = "#0F172A"
        input_bg = "rgba(255, 255, 255, 0.78)"
        button_bg = "rgba(226, 232, 240, 0.72)"
        button_hover = "rgba(147, 197, 253, 0.55)"
        shadow = "0 18px 34px rgba(15, 23, 42, 0.24)"

    html_template = """
    <script>
    (function() {
        const parentDoc = window.parent.document;
        if (!parentDoc) return;

        const rootId = "eng-calc-root";
        let root = parentDoc.getElementById(rootId);

        if (!root) {
            root = parentDoc.createElement("div");
            root.id = rootId;
            parentDoc.body.appendChild(root);
            root.innerHTML = `
                <style id="eng-calc-style">
                    #eng-calc-root {
                        --fab-bg: __FAB_BG__;
                        --panel-bg: __PANEL_BG__;
                        --panel-border: __PANEL_BORDER__;
                        --text-color: __TEXT_COLOR__;
                        --input-bg: __INPUT_BG__;
                        --button-bg: __BUTTON_BG__;
                        --button-hover: __BUTTON_HOVER__;
                        --shadow: __SHADOW__;
                    }
                    #eng-calc-fab {
                        position: fixed;
                        right: 22px;
                        bottom: 24px;
                        width: 58px;
                        height: 58px;
                        border: 1px solid rgba(255,255,255,0.25);
                        border-radius: 999px;
                        background: var(--fab-bg);
                        color: #FFFFFF;
                        font-size: 25px;
                        cursor: pointer;
                        z-index: 99990;
                        box-shadow: var(--shadow);
                    }
                    #eng-calc-fab:hover {
                        transform: translateY(-1px) scale(1.02);
                    }
                    #eng-calc-panel {
                        position: fixed;
                        right: 22px;
                        bottom: 94px;
                        width: 380px;
                        max-width: calc(100vw - 22px);
                        padding: 12px;
                        border-radius: 16px;
                        border: 1px solid var(--panel-border);
                        background: var(--panel-bg);
                        color: var(--text-color);
                        backdrop-filter: blur(12px);
                        -webkit-backdrop-filter: blur(12px);
                        box-shadow: var(--shadow);
                        z-index: 99991;
                        display: none;
                        user-select: none;
                    }
                    #eng-calc-header {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        margin-bottom: 8px;
                        cursor: move;
                        font-size: 0.92rem;
                        font-weight: 700;
                    }
                    #eng-calc-mode {
                        font-size: 0.76rem;
                        opacity: 0.9;
                        font-weight: 600;
                    }
                    #eng-calc-close {
                        background: transparent;
                        border: 0;
                        color: var(--text-color);
                        font-size: 16px;
                        cursor: pointer;
                    }
                    #eng-calc-display {
                        width: 100%;
                        box-sizing: border-box;
                        border: 1px solid var(--panel-border);
                        border-radius: 10px;
                        min-height: 46px;
                        padding: 8px 10px;
                        margin-bottom: 8px;
                        background: var(--input-bg);
                        color: var(--text-color);
                        text-align: right;
                        font-size: 1.08rem;
                    }
                    #eng-calc-grid {
                        display: grid;
                        grid-template-columns: repeat(6, 1fr);
                        gap: 7px;
                    }
                    .eng-calc-btn {
                        border: 1px solid var(--panel-border);
                        border-radius: 9px;
                        background: var(--button-bg);
                        color: var(--text-color);
                        padding: 7px 3px;
                        font-weight: 700;
                        font-size: 0.84rem;
                        cursor: pointer;
                    }
                    .eng-calc-btn:hover {
                        background: var(--button-hover);
                    }
                    .eng-calc-btn-op {
                        font-size: 0.95rem;
                    }
                    .eng-calc-btn-wide {
                        grid-column: span 2;
                    }
                    #eng-calc-actions {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 7px;
                        margin-top: 8px;
                    }
                    #eng-calc-copy,
                    #eng-calc-eq {
                        border-radius: 10px;
                        border: 1px solid var(--panel-border);
                        background: var(--fab-bg);
                        color: #FFFFFF;
                        padding: 9px 8px;
                        font-weight: 700;
                        cursor: pointer;
                    }
                </style>

                <button id="eng-calc-fab" title="Abrir calculadora científica">🖩</button>

                <div id="eng-calc-panel" aria-label="Calculadora Científica">
                    <div id="eng-calc-header">
                        <span>Calculadora Científica</span>
                        <span id="eng-calc-mode">DEG</span>
                        <button id="eng-calc-close" title="Fechar">✕</button>
                    </div>
                    <input id="eng-calc-display" type="text" value="" placeholder="0" readonly />

                    <div id="eng-calc-grid">
                        <button class="eng-calc-btn" data-ins="sin(">sin</button>
                        <button class="eng-calc-btn" data-ins="cos(">cos</button>
                        <button class="eng-calc-btn" data-ins="tan(">tan</button>
                        <button class="eng-calc-btn" data-ins="asin(">sin⁻1</button>
                        <button class="eng-calc-btn" data-ins="acos(">cos⁻1</button>
                        <button class="eng-calc-btn" data-ins="atan(">tan⁻1</button>

                        <button class="eng-calc-btn" data-ins="sinh(">sinh</button>
                        <button class="eng-calc-btn" data-ins="cosh(">cosh</button>
                        <button class="eng-calc-btn" data-ins="tanh(">tanh</button>
                        <button class="eng-calc-btn" data-ins="sqrt(">√</button>
                        <button class="eng-calc-btn" data-ins="cbrt(">∛</button>
                        <button class="eng-calc-btn" data-act="toggle-mode">DRG</button>

                        <button class="eng-calc-btn" data-ins="log(">log</button>
                        <button class="eng-calc-btn" data-ins="ln(">ln</button>
                        <button class="eng-calc-btn" data-ins="exp(">eˣ</button>
                        <button class="eng-calc-btn" data-ins="pow10(">10ˣ</button>
                        <button class="eng-calc-btn" data-ins="abs(">abs</button>
                        <button class="eng-calc-btn" data-act="eng">ENG</button>

                        <button class="eng-calc-btn" data-ins="nPr(">nPr</button>
                        <button class="eng-calc-btn" data-ins="nCr(">nCr</button>
                        <button class="eng-calc-btn" data-ins="fact(">x!</button>
                        <button class="eng-calc-btn" data-ins="^2">x²</button>
                        <button class="eng-calc-btn" data-ins="^3">x³</button>
                        <button class="eng-calc-btn" data-ins="^">xʸ</button>

                        <button class="eng-calc-btn" data-ins="(">(</button>
                        <button class="eng-calc-btn" data-ins=")">)</button>
                        <button class="eng-calc-btn" data-ins=",">,</button>
                        <button class="eng-calc-btn" data-ins="pi">π</button>
                        <button class="eng-calc-btn" data-ins="e">e</button>
                        <button class="eng-calc-btn" data-ins="ANS">Ans</button>

                        <button class="eng-calc-btn" data-ins="7">7</button>
                        <button class="eng-calc-btn" data-ins="8">8</button>
                        <button class="eng-calc-btn" data-ins="9">9</button>
                        <button class="eng-calc-btn eng-calc-btn-op" data-ins="/">÷</button>
                        <button class="eng-calc-btn" data-act="del">DEL</button>
                        <button class="eng-calc-btn" data-act="clear-all">AC</button>

                        <button class="eng-calc-btn" data-ins="4">4</button>
                        <button class="eng-calc-btn" data-ins="5">5</button>
                        <button class="eng-calc-btn" data-ins="6">6</button>
                        <button class="eng-calc-btn eng-calc-btn-op" data-ins="*">×</button>
                        <button class="eng-calc-btn" data-act="m-plus">M+</button>
                        <button class="eng-calc-btn" data-act="m-minus">M-</button>

                        <button class="eng-calc-btn" data-ins="1">1</button>
                        <button class="eng-calc-btn" data-ins="2">2</button>
                        <button class="eng-calc-btn" data-ins="3">3</button>
                        <button class="eng-calc-btn eng-calc-btn-op" data-ins="-">-</button>
                        <button class="eng-calc-btn" data-act="mr">MR</button>
                        <button class="eng-calc-btn" data-act="mc">MC</button>

                        <button class="eng-calc-btn" data-ins="0">0</button>
                        <button class="eng-calc-btn" data-ins=".">.</button>
                        <button class="eng-calc-btn" data-act="percent">%</button>
                        <button class="eng-calc-btn eng-calc-btn-op" data-ins="+">+</button>
                        <button class="eng-calc-btn eng-calc-btn-wide" data-ins="rand()">Rnd</button>
                    </div>

                    <div id="eng-calc-actions">
                        <button id="eng-calc-copy">Copiar Resultado</button>
                        <button id="eng-calc-eq">Calcular =</button>
                    </div>
                </div>
            `;

            const fab = parentDoc.getElementById("eng-calc-fab");
            const panel = parentDoc.getElementById("eng-calc-panel");
            const closeBtn = parentDoc.getElementById("eng-calc-close");
            const display = parentDoc.getElementById("eng-calc-display");
            const copyBtn = parentDoc.getElementById("eng-calc-copy");
            const eqBtn = parentDoc.getElementById("eng-calc-eq");
            const header = parentDoc.getElementById("eng-calc-header");
            const modeLabel = parentDoc.getElementById("eng-calc-mode");

            let isDeg = true;
            let ansValue = 0;
            let memoryValue = 0;

            function getDisplayValue() {
                return (display.value || "").trim();
            }

            function setDisplayValue(value) {
                display.value = String(value);
            }

            function toRad(v) {
                return isDeg ? (v * Math.PI / 180) : v;
            }

            function fromRad(v) {
                return isDeg ? (v * 180 / Math.PI) : v;
            }

            function factorial(n) {
                if (!Number.isFinite(n) || n < 0 || Math.floor(n) !== n) {
                    throw new Error("fatorial invalido");
                }
                let out = 1;
                for (let i = 2; i <= n; i += 1) out *= i;
                return out;
            }

            function nPr(n, r) {
                n = Math.floor(n);
                r = Math.floor(r);
                if (n < 0 || r < 0 || r > n) throw new Error("nPr invalido");
                let out = 1;
                for (let i = 0; i < r; i += 1) out *= (n - i);
                return out;
            }

            function nCr(n, r) {
                n = Math.floor(n);
                r = Math.floor(r);
                if (n < 0 || r < 0 || r > n) throw new Error("nCr invalido");
                r = Math.min(r, n - r);
                let num = 1;
                let den = 1;
                for (let i = 1; i <= r; i += 1) {
                    num *= (n - r + i);
                    den *= i;
                }
                return num / den;
            }

            function normalizeResult(value) {
                if (typeof value !== "number" || !isFinite(value)) return "Erro";
                const rounded = Number(value.toPrecision(12));
                return String(rounded);
            }

            function safeEvaluate(expr) {
                if (!expr) return "";
                if (!/^[A-Za-z0-9_+\-*/().,^\s]+$/.test(expr)) return "Erro";

                const jsExpr = expr.replace(/\^/g, "**");
                const scope = {
                    pi: Math.PI,
                    e: Math.E,
                    ANS: ansValue,
                    M: memoryValue,
                    sin: (x) => Math.sin(toRad(x)),
                    cos: (x) => Math.cos(toRad(x)),
                    tan: (x) => Math.tan(toRad(x)),
                    asin: (x) => fromRad(Math.asin(x)),
                    acos: (x) => fromRad(Math.acos(x)),
                    atan: (x) => fromRad(Math.atan(x)),
                    sinh: (x) => Math.sinh(x),
                    cosh: (x) => Math.cosh(x),
                    tanh: (x) => Math.tanh(x),
                    sqrt: (x) => Math.sqrt(x),
                    cbrt: (x) => Math.cbrt(x),
                    abs: (x) => Math.abs(x),
                    log: (x) => Math.log10(x),
                    ln: (x) => Math.log(x),
                    exp: (x) => Math.exp(x),
                    pow10: (x) => Math.pow(10, x),
                    fact: (x) => factorial(x),
                    nPr: (n, r) => nPr(n, r),
                    nCr: (n, r) => nCr(n, r),
                    rand: () => Math.random()
                };

                try {
                    const keys = Object.keys(scope);
                    const values = Object.values(scope);
                    const result = Function(...keys, '"use strict"; return (' + jsExpr + ');')(...values);
                    return normalizeResult(result);
                } catch (e) {
                    return "Erro";
                }
            }

            function calculateCurrent() {
                const result = safeEvaluate(getDisplayValue());
                setDisplayValue(result);
                if (result !== "Erro" && result !== "") {
                    ansValue = parseFloat(result);
                }
            }

            function insertText(text) {
                const current = getDisplayValue();
                setDisplayValue((current === "Erro" ? "" : current) + text);
            }

            function updateModeLabel() {
                modeLabel.textContent = isDeg ? "DEG" : "RAD";
            }

            fab.addEventListener("click", () => {
                panel.style.display = panel.style.display === "block" ? "none" : "block";
            });

            closeBtn.addEventListener("click", () => {
                panel.style.display = "none";
            });

            eqBtn.addEventListener("click", () => {
                calculateCurrent();
            });

            parentDoc.querySelectorAll("#eng-calc-grid .eng-calc-btn").forEach((btn) => {
                btn.addEventListener("click", () => {
                    const ins = btn.getAttribute("data-ins");
                    const act = btn.getAttribute("data-act");

                    if (ins !== null) {
                        insertText(ins);
                        return;
                    }

                    if (act === "clear-all") {
                        setDisplayValue("");
                        return;
                    }

                    if (act === "del") {
                        const current = getDisplayValue();
                        setDisplayValue(current.length ? current.slice(0, -1) : "");
                        return;
                    }

                    if (act === "toggle-mode") {
                        isDeg = !isDeg;
                        updateModeLabel();
                        return;
                    }

                    if (act === "percent") {
                        const raw = safeEvaluate(getDisplayValue());
                        if (raw === "Erro" || raw === "") {
                            setDisplayValue("Erro");
                        } else {
                            setDisplayValue(normalizeResult(parseFloat(raw) / 100));
                        }
                        return;
                    }

                    if (act === "eng") {
                        const raw = safeEvaluate(getDisplayValue());
                        if (raw === "Erro" || raw === "") {
                            setDisplayValue("Erro");
                        } else {
                            setDisplayValue(Number(parseFloat(raw)).toExponential(3));
                        }
                        return;
                    }

                    if (act === "m-plus") {
                        const raw = safeEvaluate(getDisplayValue());
                        if (raw !== "Erro" && raw !== "") memoryValue += parseFloat(raw);
                        return;
                    }

                    if (act === "m-minus") {
                        const raw = safeEvaluate(getDisplayValue());
                        if (raw !== "Erro" && raw !== "") memoryValue -= parseFloat(raw);
                        return;
                    }

                    if (act === "mr") {
                        insertText(normalizeResult(memoryValue));
                        return;
                    }

                    if (act === "mc") {
                        memoryValue = 0;
                    }
                });
            });

            copyBtn.addEventListener("click", async () => {
                const value = getDisplayValue();
                if (!value) return;
                try {
                    await window.parent.navigator.clipboard.writeText(value);
                    copyBtn.textContent = "Resultado copiado";
                } catch (e) {
                    copyBtn.textContent = "Nao foi possivel copiar";
                }
                setTimeout(() => { copyBtn.textContent = "Copiar Resultado"; }, 1200);
            });

            // Atalhos de teclado para uso rapido.
            parentDoc.addEventListener("keydown", (evt) => {
                if (panel.style.display !== "block") return;
                if (evt.key === "Enter") {
                    evt.preventDefault();
                    calculateCurrent();
                }
                if (evt.key === "Escape") {
                    panel.style.display = "none";
                }
            });

            // Drag do popup pela barra de cabecalho.
            let dragging = false;
            let offsetX = 0;
            let offsetY = 0;

            header.addEventListener("mousedown", (evt) => {
                dragging = true;
                const rect = panel.getBoundingClientRect();
                offsetX = evt.clientX - rect.left;
                offsetY = evt.clientY - rect.top;
                panel.style.right = "auto";
                panel.style.bottom = "auto";
                panel.style.left = rect.left + "px";
                panel.style.top = rect.top + "px";
            });

            parentDoc.addEventListener("mousemove", (evt) => {
                if (!dragging) return;
                panel.style.left = (evt.clientX - offsetX) + "px";
                panel.style.top = (evt.clientY - offsetY) + "px";
            });

            parentDoc.addEventListener("mouseup", () => {
                dragging = false;
            });
        }

        // Atualiza tema em rerun sem recriar elementos.
        root.style.setProperty("--fab-bg", "__FAB_BG__");
        root.style.setProperty("--panel-bg", "__PANEL_BG__");
        root.style.setProperty("--panel-border", "__PANEL_BORDER__");
        root.style.setProperty("--text-color", "__TEXT_COLOR__");
        root.style.setProperty("--input-bg", "__INPUT_BG__");
        root.style.setProperty("--button-bg", "__BUTTON_BG__");
        root.style.setProperty("--button-hover", "__BUTTON_HOVER__");
        root.style.setProperty("--shadow", "__SHADOW__");
    })();
    </script>
    """

    html = (
        html_template
        .replace("__FAB_BG__", fab_bg)
        .replace("__PANEL_BG__", panel_bg)
        .replace("__PANEL_BORDER__", panel_border)
        .replace("__TEXT_COLOR__", text_color)
        .replace("__INPUT_BG__", input_bg)
        .replace("__BUTTON_BG__", button_bg)
        .replace("__BUTTON_HOVER__", button_hover)
        .replace("__SHADOW__", shadow)
    )

    # Componente sem altura visivel; script injeta elementos no documento pai.
    components.html(html, height=0)


def init_state() -> None:
    # Inicializa chaves da sessao para evitar KeyError durante o uso.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_assistant_answer" not in st.session_state:
        st.session_state.last_assistant_answer = ""
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0
    if "chat_sidebar_collapsed" not in st.session_state:
        st.session_state.chat_sidebar_collapsed = True
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "active_model" not in st.session_state:
        st.session_state.active_model = None
    if "model_diag_error" not in st.session_state:
        st.session_state.model_diag_error = ""
    if "last_checked_api_key" not in st.session_state:
        st.session_state.last_checked_api_key = ""
    if "clipboard_image" not in st.session_state:
        st.session_state.clipboard_image = None
    if "report_history" not in st.session_state:
        st.session_state.report_history = []
    if "selected_history_idx" not in st.session_state:
        st.session_state.selected_history_idx = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "📊 Dashboard de Cálculos"
    if "theme_ui" not in st.session_state:
        st.session_state.theme_ui = "Black"
    if "current_project" not in st.session_state:
        st.session_state.current_project = "Edifício Alpha"
    if "historico" not in st.session_state:
        if "historico_conversas" in st.session_state:
            st.session_state.historico = st.session_state.historico_conversas
        else:
            st.session_state.historico = [{"titulo": "Nova Conversa", "mensagens": []}]
    if "conversa_ativa" not in st.session_state:
        if "conversa_atual_idx" in st.session_state:
            st.session_state.conversa_ativa = st.session_state.conversa_atual_idx
        else:
            st.session_state.conversa_ativa = 0
    if "confirm_clear_conversations_flag" not in st.session_state:
        st.session_state.confirm_clear_conversations_flag = False
    if "pdf_context" not in st.session_state:
        st.session_state.pdf_context = None
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = None

    # Garante que o chat principal siga a conversa ativa.
    ensure_active_conversation()
    if not st.session_state.messages and st.session_state.historico:
        st.session_state.messages = list(st.session_state.historico[st.session_state.conversa_ativa]["mensagens"])


def render_calc_module() -> None:
    # Modulo de calculos tecnicos com indicadores em tela.
    st.subheader("📊 Dashboard de Cálculos")
    st.caption("Entrada de dados técnicos → processamento Python → indicadores finais")

    # Divide tela em duas colunas: estrutural e hidraulica.
    col_left, col_right = st.columns(2)

    with col_left:
        # Bloco de dimensionamento simplificado de viga.
        st.markdown("### Dimensionamento de Vigas")
        st.latex(r"A_s = \frac{M_{sd}}{z \cdot f_{yd}}, \quad z \approx 0.9d")

        # Entradas do calculo de armadura.
        m_sd = st.number_input("Momento solicitante Msd (kN.m)", min_value=0.0, value=120.0, step=5.0, key="beam_main_msd")
        d_mm = st.number_input("Altura util d (mm)", min_value=50.0, value=450.0, step=5.0, key="beam_main_d")
        f_yd = st.number_input("Resistencia de calculo do aco fyd (MPa)", min_value=50.0, value=435.0, step=5.0, key="beam_main_fyd")

        # Calcula resultados e converte unidade para cm2 para exibicao.
        as_m2 = calc_beam_required_steel_area_m2(m_sd, d_mm, f_yd)
        as_cm2 = as_m2 * 10_000.0
        z_mm = 0.9 * d_mm

        # Botao para marcar calculo como acionado (estado de sessao).
        if st.button("Calcular Viga", use_container_width=True):
            st.session_state["beam_calculated"] = True
            add_report_to_history(
                "Dimensionamento de Viga",
                f"As={as_cm2:.2f} cm2 | z={z_mm:.1f} mm | Msd={m_sd:.1f} kN.m",
            )
        if "beam_calculated" not in st.session_state:
            st.session_state["beam_calculated"] = True

        # Mostra KPIs em tres metricas lado a lado.
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Armadura Requerida", f"{as_cm2:.2f} cm²", help="Estimativa simplificada para apoio de decisão.")
        with m2:
            st.metric("Braco de Alavanca", f"{z_mm:.1f} mm")
        with m3:
            st.metric("Momento de Entrada", f"{m_sd:.1f} kN.m")

    with col_right:
        # Bloco de hidraulica com equacao de Manning.
        st.markdown("### Hidráulica - Vazão por Manning")
        st.latex(r"Q = \frac{1}{n} A R_h^{2/3} S^{1/2}")

        # Entradas para calculo de vazao.
        area = st.number_input("Area molhada A (m2)", min_value=0.001, value=2.5, step=0.1, key="hyd_area")
        rh = st.number_input("Raio hidraulico Rh (m)", min_value=0.001, value=0.6, step=0.01, key="hyd_rh")
        slope = st.number_input("Declividade S (m/m)", min_value=0.00001, value=0.002, step=0.0001, format="%.5f", key="hyd_slope")
        n = st.number_input("Rugosidade de Manning n", min_value=0.005, value=0.015, step=0.001, format="%.3f", key="hyd_n")

        # Resultado da vazao estimada.
        q = calc_manning_flow_q(area, rh, slope, n)
        st.metric("Vazão Estimada", f"{q:.3f} m³/s")
        if st.button("Registrar Vazão no Histórico", use_container_width=True):
            add_report_to_history(
                "Hidráulica - Manning",
                f"Q={q:.3f} m3/s | A={area:.2f} m2 | Rh={rh:.2f} m | S={slope:.5f}",
            )

    # Bloco de geotecnia para indice de vazios.
    st.markdown("### Geotecnia - Índice de Vazios")
    st.latex(r"e = \frac{G_s \cdot \gamma_w}{\gamma_d} - 1")
    g1, g2, g3 = st.columns(3)
    with g1:
        gs = st.number_input("Gravidade especifica dos graos Gs", min_value=1.0, value=2.65, step=0.01, key="geo_gs")
    with g2:
        gamma_w = st.number_input("Peso especifico da agua gamma_w (kN/m3)", min_value=8.0, value=9.81, step=0.01, key="geo_gamma_w")
    with g3:
        gamma_d = st.number_input("Peso especifico seco gamma_d (kN/m3)", min_value=1.0, value=17.0, step=0.1, key="geo_gamma_d")

    # Calcula e exibe o indice de vazios.
    e = calc_void_index(gs, gamma_d, gamma_w)
    st.metric("Indice de Vazios", f"{e:.3f}")
    if st.button("Registrar Geotecnia no Histórico", use_container_width=True):
        add_report_to_history(
            "Geotecnia - Índice de Vazios",
            f"e={e:.3f} | Gs={gs:.2f} | gamma_d={gamma_d:.2f} kN/m3",
        )


def render_assistant_module(api_key: str, model_name: Optional[str], full_chat: bool = False) -> None:
    # Modulo de chat tecnico com suporte a texto + imagem + PDF.
    st.subheader("🤖 Assistente Técnico AI")
    st.caption("Suporte especializado para análise de questões de engenharia com texto, imagem e manual técnico (PDF).")

    # Garante estado consistente da conversa ativa no inicio do modulo.
    ensure_active_conversation()
    load_conversation(st.session_state.conversa_ativa)

    # Acoes rapidas do chat.
    quick_c1, quick_c2, quick_c3 = st.columns([1, 1, 1])
    with quick_c1:
        # Limpa historico da conversa.
        st.button("Limpar conversa", use_container_width=True, on_click=cb_clear_active_chat_messages)
    with quick_c2:
        # Inicia nova analise e reinicia upload.
        if st.button("Nova análise", use_container_width=True):
            st.session_state.uploader_nonce += 1
            st.session_state.clipboard_image = None
            st.rerun()
    with quick_c3:
        # Limpa PDF carregado.
        if st.session_state.pdf_context:
            if st.button("🗑️ Remover PDF", use_container_width=True):
                st.session_state.pdf_context = None
                st.session_state.pdf_filename = None
                st.rerun()

    # Botao para colar imagem do clipboard (CTRL+V).
    pasted = paste_image_button(
        "📋 Colar imagem (CTRL+V)",
        text_color="#ffffff",
        background_color="#1b2f45",
        hover_background_color="#2a4361",
        key=f"paste_btn_{st.session_state.uploader_nonce}",
    )
    # Guarda imagem colada no estado de sessao.
    if pasted.image_data is not None:
        st.session_state.clipboard_image = pasted.image_data

    # Uploader tradicional para anexar imagem.
    arquivo_colado = st.file_uploader(
        "📸 Clique aqui e dê CTRL+V para colar um print",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key=f"chat_uploader_{st.session_state.uploader_nonce}",
        label_visibility="collapsed",
    )
    st.caption("📸 Clique aqui e dê CTRL+V para colar um print")

    # Uploader de PDF para manual tecnico.
    pdf_arquivo = st.file_uploader(
        "📚 Carregar Manual Técnico (PDF)",
        type=["pdf"],
        accept_multiple_files=False,
        key=f"pdf_uploader_{st.session_state.uploader_nonce}",
        label_visibility="collapsed",
    )
    
    # Processa PDF quando carregado.
    if pdf_arquivo is not None:
        st.session_state.pdf_filename = pdf_arquivo.name
        st.session_state.pdf_context = extract_text_from_pdf(pdf_arquivo)
        st.success(f"✅ Manual carregado: {pdf_arquivo.name}")
    
    if st.session_state.pdf_context:
        st.caption(f"📚 Manual técnico ativo: {st.session_state.pdf_filename}")

    # Prioriza imagem do clipboard; se nao houver, usa upload.
    prepared_image = None
    if st.session_state.clipboard_image is not None:
        prepared_image = preprocess_pil_image(st.session_state.clipboard_image)
    elif arquivo_colado:
        prepared_image = preprocess_image(arquivo_colado)

    # Mostra preview da imagem anexada.
    if prepared_image:
        preview_c1, preview_c2 = st.columns([3, 1])
        with preview_c1:
            st.image(prepared_image.pil_image, caption="Miniatura do anexo (pré-envio)", width=260)
        with preview_c2:
            # Remove imagem atual do contexto da pergunta.
            if st.button("🗑️ Remover Imagem", use_container_width=True):
                st.session_state.clipboard_image = None
                st.session_state.uploader_nonce += 1
                st.rerun()

    # Renderiza historico da conversa.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("image"):
                st.image(message["image"], caption="Imagem enviada", use_container_width=True)

    # Entrada principal de texto do usuario.
    user_prompt = st.chat_input("Descreva o problema e o que deseja calcular...")
    if not user_prompt:
        return

    # Valida chave e modelo antes de chamar API.
    if not api_key:
        st.error("Informe a API Key na barra lateral para usar o assistente.")
        return
    if not model_name:
        st.error("Nenhum modelo Gemini ativo encontrado para a chave informada.")
        return

    # Adiciona mensagem do usuario ao historico.
    user_message = {"role": "user", "text": user_prompt}
    if prepared_image:
        user_message["image"] = prepared_image.pil_image
    st.session_state.messages.append(user_message)

    # Se for a primeira mensagem da conversa, gera titulo automatico curto.
    active_idx = st.session_state.conversa_ativa
    if active_idx is not None:
        active_chat = st.session_state.historico[active_idx]
        if active_chat["titulo"] == "Nova Conversa" and len(st.session_state.messages) == 1:
            active_chat["titulo"] = generate_chat_title(user_prompt)
    sync_messages_to_active_conversation()

    # Mostra mensagem do usuario na interface.
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if prepared_image:
            st.image(prepared_image.pil_image, caption="Imagem enviada", use_container_width=True)

    # Processa resposta do assistente com tratamento de erro.
    with st.chat_message("assistant"):
        with st.spinner("Analisando dados e resolvendo passo a passo..."):
            request_success = True
            try:
                # Chama modelo Gemini com prompt + imagem opcional + contexto PDF.
                answer = ask_gemini(api_key, model_name, user_prompt, prepared_image, st.session_state.pdf_context)
            except Exception as exc:
                request_success = False
                # Trata erro de limite de uso.
                if is_quota_error(exc):
                    st.warning("Limite temporario de uso atingido (429). Aguarde 60 segundos e tente novamente.")
                    answer = "Limite temporario de uso atingido. Aguarde 60 segundos e reenvie a pergunta."
                # Trata erro de modelo/chave invalida.
                elif is_not_found_error(exc):
                    st.error("Modelo não encontrado para esta chave. Gere uma nova chave em um novo projeto no AI Studio.")
                    answer = "Erro 404: modelo não encontrado para a chave atual."
                else:
                    # Mensagem generica para outros erros.
                    answer = f"Erro na chamada da API: {exc}"

            # Parseia resposta para extrair origem, citacoes e contexto.
            parsed = parse_assistant_response(answer)
            
            # Exibe resposta com indicador visual de origem.
            if parsed["source"] == "PDF":
                # Destaca resposta vinda do PDF.
                st.info(f"📚 **Resposta baseada no manual técnico carregado**\n\n{parsed['main_text']}")
                
                # Se houver citacao, exibe em destaque.
                if parsed["citation"]:
                    st.markdown(f"**📌 Trecho do Manual:**\n> \"{parsed['citation']}\"")
            else:
                # Aviso discreto para resposta de conhecimento geral.
                st.warning("⚠️ Informação não encontrada no manual, utilizando base global de conhecimento.")
                st.markdown(f"## Relatório Técnico\n\n{parsed['main_text']}")
            
            # Exibe detalhes de contexto em expander.
            if parsed["context"]:
                with st.expander("🔍 Detalhes da Consulta"):
                    st.markdown(parsed["context"])

    # Salva resposta no historico e no estado para uso posterior.
    st.session_state.messages.append({"role": "assistant", "text": answer})
    st.session_state.last_assistant_answer = answer
    sync_messages_to_active_conversation()
    add_report_to_history("Relatório Técnico IA", answer[:160].replace("\n", " "))

    # Limpa imagem apos envio bem-sucedido para nao reaproveitar sem querer.
    if request_success and prepared_image:
        st.session_state.clipboard_image = None
        st.session_state.uploader_nonce += 1
        st.caption("Imagem anterior limpa automaticamente para a próxima pergunta.")


def main() -> None:
    # Carrega variaveis do arquivo .env.
    load_dotenv()
    # Configura pagina Streamlit.
    st.set_page_config(page_title="Painel Engenharia Civil + IA", page_icon="🏗", layout="wide")
    # Garante inicializacao do estado da sessao.
    init_state()

    # Barra lateral com navegacao e configuracao de chave/modelo.
    with st.sidebar:
        st.markdown('<div class="sb-section-title">NAVEGAÇÃO</div>', unsafe_allow_html=True)
        st.button(
            "📊 Dashboard de Cálculos",
            use_container_width=True,
            key="nav_dashboard",
            on_click=cb_set_current_page,
            args=("📊 Dashboard de Cálculos",),
        )
        st.button(
            "💬 Assistente Técnico",
            use_container_width=True,
            key="nav_chat",
            on_click=cb_set_current_page,
            args=("💬 Assistente Técnico (Chat)",),
        )
        page = st.session_state.current_page

        st.divider()
        st.markdown('<div class="sb-section-title">CONVERSAS RECENTES</div>', unsafe_allow_html=True)
        st.caption("Somente conversas do Assistente Técnico")
        st.button("+ Nova Conversa", use_container_width=True, key="new_chat_sidebar", on_click=cb_start_new_chat)

        # Lista conversas anteriores com acao de abrir e excluir.
        for idx, conv in enumerate(st.session_state.historico):
            conv_cols = st.columns([0.82, 0.18])
            with conv_cols[0]:
                chat_title = conv.get("titulo", "Nova Conversa")
                st.button(
                    f"💬 {chat_title}",
                    use_container_width=True,
                    key=f"open_conv_{idx}",
                    on_click=cb_open_chat,
                    args=(idx,),
                )
            with conv_cols[1]:
                st.button(
                    "🗑️",
                    use_container_width=True,
                    key=f"delete_conv_{idx}",
                    type="tertiary",
                    on_click=cb_delete_chat,
                    args=(idx,),
                )

        st.button(
            "🧹 Limpar Tudo",
            use_container_width=True,
            key="clear_all_conversations_btn",
            on_click=cb_request_clear_all_chats,
        )

        if st.session_state.confirm_clear_conversations_flag:
            st.warning("Confirma apagar todo o histórico de conversas?")
            confirm_cols = st.columns(2)
            with confirm_cols[0]:
                st.button(
                    "Confirmar",
                    use_container_width=True,
                    key="confirm_clear_conversations",
                    on_click=cb_confirm_clear_all_chats,
                )
            with confirm_cols[1]:
                st.button(
                    "Cancelar",
                    use_container_width=True,
                    key="cancel_clear_conversations",
                    on_click=cb_cancel_clear_all_chats,
                )

        st.divider()
        st.markdown('<div class="sb-section-title">CONFIGURAÇÕES</div>', unsafe_allow_html=True)

        # Seletor Black/White com visual de segmentacao.
        theme_ui = st.radio("🎨 Modo Visual", ["Black", "White"], horizontal=True, key="theme_ui")

        # Projeto atual para organizar contexto e relatorios.
        st.selectbox(
            "🏗 Projeto Atual",
            ["Edifício Alpha", "Ponte Sul", "Residencial Orion", "Canal Norte"],
            key="current_project",
        )

        if st.button("📄 Salvar Relatório em PDF", use_container_width=True, key="save_pdf_btn"):
            add_report_to_history(
                "Exportação de Relatório",
                f"Projeto: {st.session_state.current_project} | Exportacao PDF solicitada.",
            )
            st.success("Solicitação de exportação para PDF registrada no histórico.")

        # Indicador visual do status do motor de calculo e versao.
        st.markdown(
            """
            <div class="sb-status">
                <span class="sb-status-dot"></span>
                <strong>Motor de Cálculo:</strong> Online<br>
                <span style="opacity:0.88;">Versão: v4.5</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Indicador de modo híbrido com PDF ativo.
        if st.session_state.pdf_context:
            st.markdown(
                """
                <div class="sb-status" style="border-color: rgba(34, 197, 94, 0.45); background: rgba(34, 197, 94, 0.12);">
                    <span class="sb-status-dot"></span>
                    <strong>📚 Modo de Consulta Híbrida Ativo</strong><br>
                    <span style="opacity:0.88;">Manual técnico carregado</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown('<div class="sb-section-title">SUPORTE</div>', unsafe_allow_html=True)
        if st.button("📘 Guia Técnico", use_container_width=True, key="support_guide"):
            st.info("Consulte a documentação técnica interna do projeto para validações finais.")
        if st.button("🛟 Abrir Suporte", use_container_width=True, key="support_contact"):
            st.info("Canal de suporte acionado. Registre sua solicitação operacional.")

        st.markdown('<div class="sb-section-title" style="margin-top:0.65rem;">HISTÓRICO DE RELATÓRIOS</div>', unsafe_allow_html=True)
        if st.session_state.report_history:
            for idx, item in enumerate(st.session_state.report_history):
                if st.button(
                    f"🗂 {item['title']} · {item['timestamp']}",
                    use_container_width=True,
                    key=f"history_item_{idx}",
                ):
                    st.session_state.selected_history_idx = idx

            selected_idx = st.session_state.selected_history_idx
            if selected_idx is not None and selected_idx < len(st.session_state.report_history):
                selected_item = st.session_state.report_history[selected_idx]
                st.caption(f"Resumo: {selected_item['summary']}")
        else:
            st.caption("Sem relatórios registrados ainda.")

        st.divider()
        st.caption("Programador Responsável")
        st.markdown("**Hélio Júnior**")

        # Tenta pegar chave de API nos secrets do Streamlit.
        secrets_key = ""
        try:
            secrets_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            secrets_key = ""

        # Se nao houver em secrets, tenta variavel de ambiente.
        env_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_key = (secrets_key or env_gemini_key).strip()

        # Mapeia escolha visual para tema da funcao de estilo.
        theme_mode = "dark" if theme_ui == "Black" else "light"

        # Revalida modelos somente quando a chave mudar.
        if gemini_key and gemini_key != st.session_state.last_checked_api_key:
            available_models, selected_model, diag_error = diagnose_and_pick_model(gemini_key)
            st.session_state.available_models = available_models
            st.session_state.active_model = selected_model
            st.session_state.model_diag_error = diag_error or ""
            st.session_state.last_checked_api_key = gemini_key

        # Chave ativa usada no assistente.
        api_key = gemini_key

    # Verifica se esta no modo chat para ajustar layout.
    is_full_chat = page == "💬 Assistente Técnico (Chat)"
    if not is_full_chat:
        st.session_state.chat_sidebar_collapsed = False

    # Aplica tema e estilos dinamicos.
    render_theme(theme_mode=theme_mode, full_chat=is_full_chat, sidebar_collapsed=st.session_state.chat_sidebar_collapsed)

    # Renderiza modulo selecionado no menu.
    if page == "📊 Dashboard de Cálculos":
        render_engineering_header()
        render_calc_module()
        render_floating_scientific_calculator(theme_mode=theme_mode)
    else:
        render_assistant_module(api_key=api_key, model_name=st.session_state.active_model, full_chat=True)

    # Rodape com aviso tecnico no dashboard.
    if page == "📊 Dashboard de Cálculos":
        st.divider()
        st.caption("Aviso: Ferramenta de apoio tecnico. Validacoes finais de projeto devem ser feitas por engenheiro responsavel.")


# Ponto de entrada do app.
if __name__ == "__main__":
    main()
