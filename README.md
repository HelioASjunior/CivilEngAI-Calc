# 🏗️ CivilEng AI-Calc v4.0

Plataforma de Engenharia Civil que combina cálculos determinísticos com IA multimodal para apoio técnico em estruturas, hidráulica, geotecnia e outras disciplinas.

---

## 🚀 Funcionalidades Principais

- 📊 Dashboard de Cálculos com módulos de estruturas, geotecnia e hidráulica.
- 💬 Assistente Técnico com foco em engenharia e normas ABNT.
- 📸 Input por print (CTRL+V) para análise de diagramas, tabelas e imagens técnicas.
- 🔐 Gerenciamento seguro de API Key via ambiente/secrets.

---

## 🧠 Base de Conhecimento Híbrida

O assistente agora utiliza uma hierarquia de consulta híbrida:

1. Prioriza o conteúdo do Manual Técnico em PDF carregado/local.
2. Se o PDF não cobrir o tema, utiliza conhecimento global de engenharia e normas ABNT.
3. Em caso de conflito de diretriz, o Manual Técnico do projeto é priorizado.

---

## 🖩 Calculadora Científica Integrada

O dashboard inclui uma calculadora científica flutuante para cálculos rápidos, com:

- Operações básicas: `+`, `-`, `*`, `/`
- Trigonometria: `sin`, `cos`, `tan` e inversas
- Potência e raízes: `x^y`, `x²`, `x³`, `√`, `∛`
- Logaritmos: `log`, `ln`, `10^x`, `e^x`
- Recursos extras: memória (`M+`, `M-`, `MR`, `MC`), `nPr`, `nCr`, `%`, `Ans`
- Botão de cópia de resultado para colar no chat ou nos campos do dashboard

---

## 🔎 Rastreabilidade de Fontes

O chat indica visualmente a origem da resposta:

- Manual Técnico (PDF): resposta marcada como fonte do manual.
- Base Global: aviso de fallback quando a informação não estiver no manual.

Também há seção de detalhes de consulta para auditoria do contexto utilizado na resposta.

---

## 🛠️ Tecnologias Utilizadas

- Python 3.10+
- Streamlit
- Google GenAI SDK v1 (Gemini 1.5 Flash)
- Pillow (PIL)
- PyPDF2

---

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/calc-eng-civil.git
cd calc-eng-civil
```

2. Crie e ative um ambiente virtual (recomendado).

3. Instale as dependências (inclui `PyPDF2`):

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuração da API

1. Gere uma chave no Google AI Studio.
2. Defina `GEMINI_API_KEY` no `.env` da raiz do projeto ou em `st.secrets`.

---

## 📚 Guia de Uso do PDF

1. Crie a pasta `/documentos` na raiz do projeto (se ainda não existir).
2. Coloque o Manual Técnico em PDF nessa pasta para carregamento automático.
3. No chat, confirme que o modo híbrido está ativo antes de iniciar as consultas.

---

## ▶️ Execução

```bash
streamlit run app.py
```

---

## ⚠️ Nota de Engenharia

Ferramenta de apoio técnico. Toda decisão final deve ser validada por engenheiro responsável e conferida com as normas ABNT aplicáveis.