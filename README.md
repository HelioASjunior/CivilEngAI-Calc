# 🏗️ CivilEng AI-Calc v4.0

Uma plataforma avançada de **Engenharia Civil** que combina o rigor dos cálculos normativos em Python com a flexibilidade da Inteligência Artificial Multimodal do Google (Gemini 1.5 Flash).


---

## 🚀 Funcionalidades Principais

* **📊 Dashboard de Cálculos:** Módulos de cálculo determinístico para estruturas (flexão, cisalhamento), geotecnia e hidráulica.
* **💬 Assistente Técnico (IA):** Chat especializado treinado para interpretar normas brasileiras (NBRs).
* **📸 Input por Print (CTRL+V):** Suporte exclusivo para colar diagramas do Ftool, AutoCAD ou fotos de obra diretamente no chat para análise instantânea.
* **🌓 Interface Adaptativa:** Modos Dark (Black) e Light (White) com correção automática de contraste para fórmulas em LaTeX.
* **🔐 Segurança de Dados:** Gerenciamento de API Key via interface ou segredos de ambiente, sem exposição de credenciais no código.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Interface:** [Streamlit](https://streamlit.io/)
* **IA Multimodal:** [Google GenAI SDK v1](https://pypi.org/project/google-genai/) (Gemini 1.5 Flash)
* **Processamento de Imagem:** Pillow (PIL)
* **Matemática/Fórmulas:** NumPy & KaTeX (LaTeX)

---

## 📦 Como Instalar e Rodar

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/calc-eng-civil.git](https://github.com/seu-usuario/calc-eng-civil.git)
   cd calc-eng-civil
Instale as dependências:
(Recomenda-se o uso de um ambiente virtual)

Bash
pip install -U google-genai streamlit Pillow pandas numpy
Configure sua API Key:

Obtenha uma chave gratuita no Google AI Studio.

Crie um arquivo .env na raiz do projeto ou insira a chave diretamente na barra lateral do aplicativo.

Execute a aplicação:

Bash
streamlit run app.py


⚠️ Notas de Engenharia
Este software deve ser utilizado como uma ferramenta de auxílio. Todos os resultados gerados pela IA devem ser conferidos com os cálculos determinísticos apresentados no Dashboard e validados por um engenheiro responsável, conforme as normas da ABNT.