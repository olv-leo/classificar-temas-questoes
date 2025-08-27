import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from PIL import Image
import pytesseract
import pandas as pd
from google import genai

# ── Setup ────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("processamento.log"), logging.StreamHandler()],
)

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

BASE_PATH = Path(os.getenv("BASE_PATH"))
EXTENSOES = tuple(map(str.lower, os.getenv("EXTENSOES").split(",")))
ARQUIVO_SAIDA = Path("questoes_classificadas.xlsx")

ARQUIVO_BD_QUESTOES = Path("bd_questoes.xlsx")
ARQUIVO_LISTA_ASSUNTOS = Path("lista_assuntos.xlsx")

# Anos e semestres fixos
ANOS_SEMESTRES = [
    {"ano": 2015, "semestre": 1},
    {"ano": 2015, "semestre": 2},
    {"ano": 2016, "semestre": 1},
    {"ano": 2016, "semestre": 2},
    {"ano": 2017, "semestre": 1},
    {"ano": 2017, "semestre": 2},
    {"ano": 2018, "semestre": 1},
    {"ano": 2018, "semestre": 2},
    {"ano": 2019, "semestre": 1},
    {"ano": 2019, "semestre": 2},
    {"ano": 2020, "semestre": 1},
    {"ano": 2022, "semestre": 2},
    {"ano": 2023, "semestre": 1},
    {"ano": 2023, "semestre": 2},
    {"ano": 2024, "semestre": 1},
    {"ano": 2024, "semestre": 2},
    {"ano": 2025, "semestre": 2},
]

logging.info("Iniciando script de classificação de questões")

# ── Carregamentos ──────────────────────────────────────────────────────────────
try:
    tabela_assuntos = pd.read_excel(ARQUIVO_LISTA_ASSUNTOS)
    logging.info(f"Lista de assuntos carregada ({len(tabela_assuntos)} tópicos)")
except Exception as e:
    logging.error(f"Erro ao carregar {ARQUIVO_LISTA_ASSUNTOS}: {e}")
    raise

try:
    bd_questoes = pd.read_excel(ARQUIVO_BD_QUESTOES)
    logging.info(f"BD de questões carregado ({len(bd_questoes)} registros)")
except Exception as e:
    logging.error(f"Erro ao carregar {ARQUIVO_BD_QUESTOES}: {e}")
    raise


# ── Seleções via JSON ─────────────────────────────────────────────────────────
def carregar_selecoes() -> Dict[Tuple[int, int], Set[str]]:
    """
    Carrega seleções de um arquivo local 'selecoes.json', se existir.
    Estrutura: [{ano, semestre, questao}, ...]
    """
    p = Path("selecoes.json")
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) == 0:
            return {}
        selecoes: Dict[Tuple[int, int], Set[str]] = {}
        for item in data:
            ano = int(item["ano"])
            semestre = int(item["semestre"])
            # força número e converte de volta para string sem zeros à esquerda
            questao = str(int(item["questao"]))
            selecoes.setdefault((ano, semestre), set()).add(questao)
        return selecoes
    except Exception as e:
        logging.error(f"Erro ao ler selecoes.json: {e}")
        return {}


SELECOES = carregar_selecoes()


# ── Funções principais ────────────────────────────────────────────────────────
def processar_resposta_gemini(
    resposta: str, numero_questao: str
) -> List[Dict[str, str]]:
    linhas_classificadas = []
    for linha in resposta.strip().split("\n"):
        linha = linha.strip()
        if not linha or "|" not in linha or linha.startswith(("| tema", "|---")):
            continue
        partes = [p.strip() for p in linha.split("|")]

        if len(partes) != 5:
            logging.warning(f"Partes: {partes}")
            raise ValueError(
                f"Resposta inesperada para Q{numero_questao}: '{linha}' (esperado 5 colunas, obteve {len(partes)})"
            )

        tema = partes[2]
        topico = partes[3]
        if tema and topico:
            linhas_classificadas.append(
                {"numero_questao": numero_questao, "tema": tema, "topico": topico}
            )
    return linhas_classificadas


def obter_info_questao(numero_questao: str, ano: int, semestre: int):
    try:
        m = bd_questoes[
            (bd_questoes["numero_questao"] == int(numero_questao))
            & (bd_questoes["ano"] == int(ano))
            & (bd_questoes["semestre"] == int(semestre))
        ]
        if not m.empty:
            return m.iloc[0]["materia"], m.iloc[0]["gabarito"]
        logging.warning(
            f"Questão {numero_questao} ({ano}/{semestre}) não encontrada no BD"
        )
        return "N/A", "N/A"
    except Exception as e:
        logging.error(f"Erro ao buscar questão {numero_questao} no BD: {e}")
        return "N/A", "N/A"


def classificar_questao(
    numero_questao: str, texto_questao: str, ano: int, semestre: int
):
    try:
        materia_questao, gabarito_questao = obter_info_questao(
            numero_questao, ano, semestre
        )
        if materia_questao == "N/A":
            topicos_prioritarios = tabela_assuntos
        else:
            topicos_prioritarios = tabela_assuntos[
                tabela_assuntos["materia"] == materia_questao
            ]

        assuntos_str = pd.concat(
            [
                topicos_prioritarios,
                tabela_assuntos[
                    ~tabela_assuntos.index.isin(topicos_prioritarios.index)
                ],
            ]
        ).to_csv(index=False, sep="|")

        prompt = f"""
Você deve se comportar como um professor de ensino médio que tem que classificar os assuntos que cada uma das questões enviadas aborda. 
Você deve analisar o conteúdo da questão e avaliar quais assuntos da lista de assuntos estão presentes na questão. 
Vou te enviar o texto da questão e você deve responder apenas com os temas e tópicos da lista de assuntos que a questão aborda.
Sua resposta deve estar no formato de tabela com as colunas: 
| tema | topico 

Sua resposta não deve conter nenhuma outra informação além da tabela.
Sua resposta não deve conter `````, nem qualquer outra coisa que não seja a tabela.
Exemplo de resposta correta:
| tema | topico |
|---|---|
| Grandezas físicas | Medição de tempo |
| Eletrodinâmica | Corrente elétrica |
| Eletrostática | Carga elétrica |

Uma questão pode abordar mais de um tópico, nesse caso cada tópico deve ser uma linha. Você não deve inventar novos tópicos deve usar apenas os tópicos presentes na lista. 
Você deve usar apenas as colunas tema e topico da lista de assuntos.
A lista de assuntos é:
{assuntos_str}

Questão:
{texto_questao}
""".strip()
        logging.info(
            f"Enviando Q{numero_questao} ({ano}/{semestre}) para classificação"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        print(response.text)
        classificacoes = processar_resposta_gemini(response.text, numero_questao)
        return classificacoes, materia_questao, gabarito_questao
    except Exception as e:
        logging.error(f"Erro ao classificar questão {numero_questao}: {e}")
        return (
            [
                {
                    "numero_questao": numero_questao,
                    "tema": "ERRO",
                    "topico": f"Erro: {str(e)}",
                }
            ],
            "ERRO",
            "ERRO",
        )


def construir_caminho_pasta(ano: int, semestre: int) -> Path:
    return BASE_PATH / str(ano) / f"{semestre}º Semestre" / "02-Fotos Questões" / "PNG"


def listar_arquivos_para_processar(
    pasta: Path, ano: int, semestre: int, selecoes: Dict[Tuple[int, int], Set[str]]
) -> List[str]:
    if not pasta.exists():
        logging.info("Pasta não encontrada")
        return []

    # Lista todos os arquivos válidos
    todos = [
        f
        for f in os.listdir(pasta)
        if f.lower().endswith(EXTENSOES) and (pasta / f).is_file()
    ]

    # Há seleções globais em selecoes.json?
    ha_selecoes_globais = any(len(s) > 0 for s in selecoes.values())

    # Seleções específicas para este (ano, semestre)
    chaves = selecoes.get((ano, semestre), set())

    # Se há seleções globais e este (ano, semestre) não está nelas -> ignorar totalmente
    if ha_selecoes_globais and not chaves:
        logging.info(
            f"Ignorando {ano}/{semestre} porque não está no selecoes.json (modo seleção ativo)."
        )
        return []

    # Sem seleções globais -> processa tudo (comportamento normal)
    if not ha_selecoes_globais:
        logging.info(
            f"Processando todas as questões de {ano}/{semestre} (sem seleções definidas)."
        )
        return todos

    # Modo seleção ativo E existem chaves para este (ano, semestre):
    # processar apenas as questões listadas
    def normaliza_nome(stem: str) -> str:
        return str(int(stem)) if stem.isdigit() else stem.strip()

    filtrados: List[str] = []
    for f in todos:
        nome = Path(f).stem
        if normaliza_nome(nome) in chaves:
            filtrados.append(f)

    if not filtrados:
        logging.info(
            f"Nenhum arquivo correspondente às seleções {sorted(chaves)} em {ano}/{semestre}."
        )
    else:
        logging.info(
            f"Processando {len(filtrados)} questão(ões) selecionada(s) em {ano}/{semestre}: {sorted(chaves)}"
        )

    return filtrados


def processar_ano_semestre(ano: int, semestre: int):
    pasta = construir_caminho_pasta(ano, semestre)
    if not pasta.exists():
        logging.warning(f"Pasta não encontrada: {pasta}")
        return

    arquivos = listar_arquivos_para_processar(pasta, ano, semestre, SELECOES)
    logging.info(f"Processando {ano}/{semestre} - {len(arquivos)} arquivo(s)")

    for arquivo in arquivos:
        caminho = pasta / arquivo
        try:
            numero_questao = Path(arquivo).stem
            imagem = Image.open(caminho)
            texto = pytesseract.image_to_string(imagem, lang="por")

            classificacoes, materia, gabarito = classificar_questao(
                numero_questao, texto, ano, semestre
            )

            linhas = []
            for c in classificacoes:
                linhas.append(
                    {
                        "arquivo": arquivo,
                        "caminho_completo": str(caminho),
                        "numero_questao": c["numero_questao"],
                        "texto_questao": texto,
                        "tema": c["tema"],
                        "topico": c["topico"],
                        "ano": ano,
                        "semestre": semestre,
                        "materia": materia,
                        "gabarito": gabarito,
                    }
                )

            nova_linha = pd.DataFrame(linhas)

            if ARQUIVO_SAIDA.exists():
                df_existente = pd.read_excel(ARQUIVO_SAIDA)
                df_completo = pd.concat([df_existente, nova_linha], ignore_index=True)
            else:
                df_completo = nova_linha

            df_completo.to_excel(ARQUIVO_SAIDA, index=False)
            logging.info(
                f"Q{numero_questao} ({ano}/{semestre}) salva em {ARQUIVO_SAIDA}"
            )

        except Exception as e:
            logging.error(f"Erro ao processar {arquivo}: {e}")


# ── Loop principal ────────────────────────────────────────────────────────────
for cfg in ANOS_SEMESTRES:
    processar_ano_semestre(int(cfg["ano"]), int(cfg["semestre"]))

logging.info("Processamento concluído")
