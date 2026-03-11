import io
import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image, ImageOps
from streamlit_paste_button import paste_image_button


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


def build_engineering_system_instruction() -> str:
    # Instrucao de sistema fixa para orientar o modelo em respostas tecnicas.
    return (
        "Voce e um calculista estrutural. "
        "Responda com precisao matematica, cite a NBR 6118 e use unidades do SI. "
        "Formate a resposta como relatorio tecnico em Markdown, incluindo: "
        "(1) resumo do problema, (2) premissas adotadas, (3) tabela de dados de entrada, "
        "(4) formulas em LaTeX, (5) calculo passo a passo, (6) conclusao objetiva."
    )


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

    # Prioriza modelo Flash quando estiver disponivel.
    flash_candidates = [m for m in available_models if "flash" in m.lower()]
    selected_model = flash_candidates[0] if flash_candidates else available_models[0]
    return available_models, selected_model, None


def is_not_found_error(exc: Exception) -> bool:
    # Verifica se o erro e de recurso/modelo nao encontrado.
    message = str(exc).upper()
    return "404" in message or "NOT_FOUND" in message


def ask_gemini(api_key: str, model_name: str, prompt: str, prepared_image: Optional[PreparedImage]) -> str:
    # Cria cliente Gemini.
    client = genai.Client(api_key=api_key)

    # Conteudo base enviado ao modelo (texto do usuario).
    contents = [prompt]
    # Se houver imagem, envia junto com o texto.
    if prepared_image:
        contents.append(prepared_image.pil_image)

    # Gera resposta com instrucao de sistema tecnica.
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config={"system_instruction": build_engineering_system_instruction()},
    )
    # Retorna texto da resposta ou fallback padrao.
    return response.text if getattr(response, "text", None) else "Sem resposta da API."


def is_quota_error(exc: Exception) -> bool:
    # Verifica se o erro indica limite de uso/quota esgotada.
    message = str(exc).upper()
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "RESOURCE EXHAUSTED" in message


# Funcoes de interface (Streamlit).
# ---------- UI ----------
def render_theme(theme_mode: str = "light", full_chat: bool = False, sidebar_collapsed: bool = False) -> None:
    # Ajustes de layout para modo normal ou chat em tela mais ampla.
    top_padding = "0.6rem" if full_chat else "1.2rem"
    max_width = "1380px" if full_chat else "1200px"
    full_chat_css = "[data-testid=\"stVerticalBlock\"]:has([data-testid=\"stChatMessage\"]) {padding-top: 0.2rem;}" if full_chat else ""
    collapsed_sidebar_css = """
    [data-testid="stSidebar"] {margin-left: -22rem;}
    [data-testid="stSidebarCollapsedControl"] {display: none;}
    """ if (full_chat and sidebar_collapsed) else ""

    # Define paleta de cores para tema escuro ou claro.
    is_dark = theme_mode == "dark"
    if is_dark:
        app_bg = "linear-gradient(180deg, #0d1726 0%, #1a2635 100%)"
        base_text = "#e8edf5"
        sidebar_bg = "linear-gradient(180deg, #0b1220 0%, #16263a 100%)"
        card_bg = "#162234"
        card_border = "#2f435d"
        tab_bg = "#1f3046"
        tab_border = "#3a536f"
        tab_text = "#e8edf5"
        tab_active_bg = "#0f1c2d"
        input_bg = "#101a28"
        input_text = "#edf2f9"
        input_border = "#4f6885"
        chat_bg = "#121f2f"
    else:
        app_bg = "linear-gradient(180deg, #f4f6f8 0%, #ffffff 100%)"
        base_text = "#0f1724"
        sidebar_bg = "linear-gradient(180deg, #0f2842 0%, #1e3d5a 100%)"
        card_bg = "#ffffff"
        card_border = "#d7dfe7"
        tab_bg = "#ecf1f6"
        tab_border = "#cbd7e4"
        tab_text = "#112338"
        tab_active_bg = "#153453"
        input_bg = "#ffffff"
        input_text = "#0f1724"
        input_border = "#8fa4bc"
        chat_bg = "#f8fbff"

    # CSS base da aplicacao com variaveis substituidas dinamicamente.
    css = """
        <style>
            :root {
                --eng-base-text: __BASE_TEXT__;
                --eng-card-bg: __CARD_BG__;
            }
            .stApp {
                background: __APP_BG__;
                color: var(--eng-base-text);
                transition: background 0.35s ease, color 0.35s ease;
            }
            [data-testid="stSidebar"] {
                background: __SIDEBAR_BG__;
                border-right: 1px solid __CARD_BORDER__;
                transition: background 0.35s ease;
            }
            [data-testid="stSidebar"] * {
                color: #ffffff;
            }
            h1, h2, h3, h4, h5, h6, p, label, span, li, div {
                color: var(--eng-base-text);
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] h5,
            [data-testid="stSidebar"] h6,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] li,
            [data-testid="stSidebar"] div {
                color: #ffffff;
            }
            .block-container {
                padding-top: __TOP_PADDING__;
                max-width: __MAX_WIDTH__;
            }
            .eng-card {
                background: __CARD_BG__;
                border: 1px solid __CARD_BORDER__;
                border-radius: 14px;
                padding: 1rem 1.2rem;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                transition: background 0.35s ease, border-color 0.35s ease;
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
                color: #ffffff !important;
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
                transition: background 0.35s ease, color 0.35s ease, border-color 0.35s ease;
            }
            [data-testid="stFileUploaderDropzone"] {
                border-width: 1px !important;
                border-style: solid !important;
                border-radius: 10px !important;
                padding: 0.55rem 0.7rem !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"] {
                font-size: 0.82rem !important;
                opacity: 0.9;
            }
            .stButton > button,
            .stDownloadButton > button,
            .stSlider [role="slider"] {
                border: 1px solid __INPUT_BORDER__ !important;
            }
            [data-testid="stChatMessage"] {
                background: __CHAT_BG__;
                border: 1px solid __CARD_BORDER__;
                border-radius: 10px;
                padding: 0.2rem 0.7rem;
            }
            .katex,
            .katex * {
                color: var(--eng-base-text) !important;
            }
            .stDataFrame, .stTable, .stDataFrame *, .stTable * {
                color: var(--eng-base-text) !important;
                background-color: transparent !important;
            }
            table, th, td {
                color: var(--eng-base-text) !important;
                border-color: __CARD_BORDER__ !important;
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
        .replace("__CHAT_BG__", chat_bg)
        .replace("__FULL_CHAT_CSS__", full_chat_css)
        .replace("__COLLAPSED_SIDEBAR_CSS__", collapsed_sidebar_css)
    )

    # Renderiza CSS no app.
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


def render_engineering_header() -> None:
    # Cabecalho principal da pagina de calculos.
    st.markdown("""
    <div class="eng-card">
        <h2 style="margin-bottom:0.3rem;">Plataforma Profissional de Engenharia Civil</h2>
        <p style="margin-top:0;">Entrada tecnica, processamento matematico e suporte inteligente multimodal para decisoes de projeto.</p>
    </div>
    """, unsafe_allow_html=True)


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

        # Botao para marcar calculo como acionado (estado de sessao).
        if st.button("Calcular Viga", use_container_width=True):
            st.session_state["beam_calculated"] = True
        if "beam_calculated" not in st.session_state:
            st.session_state["beam_calculated"] = True

        # Calcula resultados e converte unidade para cm2 para exibicao.
        as_m2 = calc_beam_required_steel_area_m2(m_sd, d_mm, f_yd)
        as_cm2 = as_m2 * 10_000.0
        z_mm = 0.9 * d_mm
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


def render_assistant_module(api_key: str, model_name: Optional[str], full_chat: bool = False) -> None:
    # Modulo de chat tecnico com suporte a texto + imagem.
    st.subheader("💬 Assistente Técnico")
    st.caption("Suporte especializado para análise de questões de engenharia com texto e imagem.")

    # Acoes rapidas do chat.
    quick_c1, quick_c2 = st.columns([1, 1])
    with quick_c1:
        # Limpa historico da conversa.
        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_assistant_answer = ""
            st.rerun()
    with quick_c2:
        # Inicia nova analise e reinicia upload.
        if st.button("Nova análise", use_container_width=True):
            st.session_state.uploader_nonce += 1
            st.session_state.clipboard_image = None
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
                # Chama modelo Gemini com prompt + imagem opcional.
                answer = ask_gemini(api_key, model_name, user_prompt, prepared_image)
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

            # Exibe resposta em formato de relatorio tecnico.
            st.markdown(f"## Relatório Técnico\n\n{answer}")

    # Salva resposta no historico e no estado para uso posterior.
    st.session_state.messages.append({"role": "assistant", "text": answer})
    st.session_state.last_assistant_answer = answer

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
        st.header("Navegação")

        # Seletor de modulo principal.
        page = st.radio("Seções", ["📊 Dashboard de Cálculos", "💬 Assistente Técnico (Chat)"])

        # Tenta pegar chave de API nos secrets do Streamlit.
        secrets_key = ""
        try:
            secrets_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            secrets_key = ""

        # Se nao houver em secrets, tenta variavel de ambiente.
        env_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_key = (secrets_key or env_gemini_key).strip()

        # Tema atualmente fixo em dark.
        theme_mode = "dark"

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
    else:
        render_assistant_module(api_key=api_key, model_name=st.session_state.active_model, full_chat=True)

    # Rodape com aviso tecnico no dashboard.
    if page == "📊 Dashboard de Cálculos":
        st.divider()
        st.caption("Aviso: Ferramenta de apoio tecnico. Validacoes finais de projeto devem ser feitas por engenheiro responsavel.")


# Ponto de entrada do app.
if __name__ == "__main__":
    main()
