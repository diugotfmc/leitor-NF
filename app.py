# -*- coding: utf-8 -*-
import io
import re
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd

import fitz  # PyMuPDF


# ===============================
# Helpers de números (pt-BR)
# ===============================
def to_float_br(s: str) -> Optional[float]:
    """
    Converte string de número no formato BR (1.234,56) para float.
    Retorna None caso não seja número válido.
    """
    if s is None:
        return None
    s = s.strip()
    # aceita 1.234,56  ou 1234,56  ou 1.234.567,00
    if not re.search(r"\d+[\.,]?\d*", s):
        return None
    # se tem vírgula, assume que vírgula é decimal
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def to_int_safe(x) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


# ===============================
# Extração de texto do PDF
# ===============================
def pdf_to_pages_text(file_bytes: bytes) -> List[str]:
    """
    Lê o PDF e retorna uma lista de textos, um por página.
    """
    pages_text = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            pages_text.append(page.get_text("text"))
    return pages_text


def pdf_to_pages_words(file_bytes: bytes) -> List[List[Tuple[float, float, float, float, str]]]:
    """
    Retorna, por página, a lista de (x0, y0, x1, y1, texto) para cada palavra.
    Útil para o modo "Avançado".
    """
    all_pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            words = page.get_text("words")  # x0, y0, x1, y1, "text", block, line, word_no
            words = [(w[0], w[1], w[2], w[3], w[4]) for w in words]
            # ordena por y e depois x
            words.sort(key=lambda w: (round(w[1], 1), w[0]))
            all_pages.append(words)
    return all_pages


# ===============================
# Delimitadores de seção
# ===============================
START_MARKERS = [
    "DADOS DO PRODUTO / SERVIÇO",
    "DADOS DO PRODUTO/SERVIÇO",
    "DADOS DO PRODUTO / SERVICO",  # sem acento
]
END_MARKERS = [
    "DADOS ADICIONAIS",
    "INFORMAÇÕES COMPLEMENTARES",
    "INFORMACOES COMPLEMENTARES",
    "DADOS ADICIONAIS (COMPLEMENTO)",
    "INFORMAÇÕES ADICIONAIS",
    "INFORMACOES ADICIONAIS"
]


def recorta_tabela_itens_por_texto(pages_text: List[str]) -> str:
    """
    Recorta o bloco de texto correspondente à tabela de itens,
    do marcador START até o marcador END (por página).
    Concatena em um único texto.
    """
    blobs = []
    for txt in pages_text:
        lower = txt.upper()
        ini = None
        for m in START_MARKERS:
            pos = lower.find(m)
            if pos != -1:
                ini = pos + len(m)
                break
        if ini is None:
            continue
        fim = None
        for m in END_MARKERS:
            pos = lower.find(m, ini)
            if pos != -1:
                fim = pos
                break
        if fim is None:
            fim = len(txt)
        blobs.append(txt[ini:fim])
    return "\n".join(blobs)


# ===============================
# Parser (modo Padrão - regex)
# ===============================
RE_ITEM_START = re.compile(r"\b([A-Z]{2}[A-Z0-9]{2,})\b")  # ex.: AC0302BJ02800629
RE_IT_CODE = re.compile(r"\bIT\d{2,}\b", re.IGNORECASE)
RE_NM_CODE = re.compile(r"\bNM\d{5,}\b", re.IGNORECASE)
RE_NCM = re.compile(r"\b\d{8}\b")
RE_CFOP = re.compile(r"\b\d{4}\b")
RE_QTD_UN = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2,4})\s+([A-Z]{2,3})\b")  # 205,0000 KG | 1,0000 UN
RE_MONEY = re.compile(r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b")


def split_blocos_por_item(tabela_texto: str) -> List[str]:
    """
    Divide o texto da tabela em blocos de item, usando início de código de produto como âncora.
    """
    lines = [re.sub(r"\s+", " ", l).strip() for l in tabela_texto.splitlines() if l.strip()]
    # Junta linhas, pois alguns itens quebram em várias linhas
    joined = " ".join(lines)
    # Força delimitador antes de cada possível código de item
    marked = RE_ITEM_START.sub(r"\n\1", joined).strip()
    itens = [i.strip() for i in marked.split("\n") if i.strip()]
    # pode vir um cabeçalho solto antes do 1º item; remove-o se não for item
    if itens and not RE_ITEM_START.match(itens[0]):
        itens = itens[1:]
    return itens


def extrai_campos_item_por_regex(bloco: str) -> Dict[str, Optional[str]]:
    """
    Extrai campos principais do texto de um item usando heurísticas.
    Pensado para DANFE com colunas: CÓD PROD | DESCRIÇÃO | NCM | CFOP | UN | QTD | V.UNIT | V.TOTAL
    """
    d: Dict[str, Optional[str]] = {
        "codigo_produto": None,
        "it_code": None,
        "nm_code": None,
        "descricao": None,
        "ncm": None,
        "cfop": None,
        "un": None,
        "qtd": None,
        "valor_unitario": None,
        "valor_total": None,
    }

    # 1) Código de produto
    m = RE_ITEM_START.search(bloco)
    if not m:
        return d
    d["codigo_produto"] = m.group(1)

    # 2) IT / NM
    it = RE_IT_CODE.search(bloco)
    nm = RE_NM_CODE.search(bloco)
    d["it_code"] = it.group(0) if it else None
    d["nm_code"] = nm.group(0) if nm else None

    # 3) NCM e CFOP (o '000' às vezes aparece antes; pula CFOP inválido)
    ncm = RE_NCM.search(bloco)
    if ncm:
        d["ncm"] = ncm.group(0)
        # busca CFOP após NCM
        tail = bloco[ncm.end():]
        cfops = RE_CFOP.findall(tail)
        cfop = None
        for c in cfops:
            if c != "000":
                cfop = c
                break
        d["cfop"] = cfop

    # 4) Descrição: trecho entre NM... e NCM
    if nm and ncm:
        desc = bloco[nm.end():ncm.start()]
        desc = re.sub(r"^\s*[-–]\s*", "", desc)
        d["descricao"] = desc.strip(" -")
    else:
        # fallback: entre codigo_produto e NCM
        if d["codigo_produto"] and ncm:
            desc = bloco[m.end():ncm.start()]
            desc = desc.replace(" - ", " ").strip(" -")
            d["descricao"] = desc.strip() or None

    # 5) Quantidade & Unidade: usa a ÚLTIMA ocorrência (ex.: "1,0000 UN" | "205,0000 KG")
    qmatches = list(RE_QTD_UN.finditer(bloco))
    if qmatches:
        q, un = qmatches[-1].groups()
        d["qtd"] = q
        d["un"] = un

    # 6) Valores monetários antes do trecho "FCP do ICMS" (evita valores do complemento)
    corte_fcp = bloco.find("FCP DO ICMS")
    bloco_val = bloco if corte_fcp == -1 else bloco[:corte_fcp]
    money = [m.group(0) for m in RE_MONEY.finditer(bloco_val)]

    # Heurística: valor_total costuma ser o MAIOR valor antes do FCP.
    # valor_unitario = valor_total / qtd (quando qtd válida e > 0).
    vtotal = None
    if money:
        # ignora zeros tipo "0,00" que confundem a ordenação de unitário
        nvals = [to_float_br(x) for x in money if to_float_br(x) is not None]
        if nvals:
            vtotal = max(nvals)
    if vtotal is not None:
        d["valor_total"] = f"{vtotal:.2f}".replace(".", ",")
        if d["qtd"]:
            qv = to_float_br(d["qtd"])
            if qv and qv > 0:
                vunit = vtotal / qv
                d["valor_unitario"] = f"{vunit:.2f}".replace(".", ",")
    return d


def extrair_itens_modo_padrao(pages_text: List[str]) -> pd.DataFrame:
    """
    Modo padrão: text mining via regex/heurística.
    Retorna DataFrame com colunas principais.
    """
    bloco = recorta_tabela_itens_por_texto(pages_text)
    itens = split_blocos_por_item(bloco)
    registros = [extrai_campos_item_por_regex(b) for b in itens]
    df = pd.DataFrame(registros)

    # Normaliza numéricos extra (opcional)
    for col in ["qtd", "valor_unitario", "valor_total"]:
        df[f"{col}_num"] = df[col].apply(to_float_br)
    # Ordena por IT se existir
    if "it_code" in df.columns and df["it_code"].notna().any():
        df["_itnum"] = df["it_code"].str.extract(r"(\d+)", expand=False).apply(to_int_safe)
        df = df.sort_values(by=["_itnum", "codigo_produto"], kind="stable").drop(columns=["_itnum"])
    return df


# ===============================
# Parser (modo Avançado - coordenadas)
# ===============================
def agrupa_linhas_por_y(words: List[Tuple[float, float, float, float, str]], tol: float = 2.0) -> List[List[Tuple]]:
    """
    Agrupa palavras por linha usando proximidade de Y.
    """
    linhas = []
    atual_y = None
    atual = []
    for (x0, y0, x1, y1, t) in words:
        if atual_y is None:
            atual_y = y0
            atual = [(x0, y0, x1, y1, t)]
            continue
        if abs(y0 - atual_y) <= tol:
            atual.append((x0, y0, x1, y1, t))
        else:
            linhas.append(sorted(atual, key=lambda z: z[0]))
            atual = [(x0, y0, x1, y1, t)]
            atual_y = y0
    if atual:
        linhas.append(sorted(atual, key=lambda z: z[0]))
    return linhas


def extrair_itens_modo_avancado(pages_words: List[List[Tuple[float, float, float, float, str]]]) -> pd.DataFrame:
    """
    Modo avançado: reconstrói linhas pela coordenada Y e extrai por regex da linha.
    Mantém a heurística do modo padrão para os campos.
    """
    linhas_texto = []
    for words in pages_words:
        linhas = agrupa_linhas_por_y(words)
        # seleciona apenas linhas que parecem pertencer à tabela (possuem NCM de 8 dígitos OU CFOP de 4 dígitos)
        for ln in linhas:
            txt = " ".join([w[4] for w in ln])
            if RE_NCM.search(txt) or RE_CFOP.search(txt):
                linhas_texto.append(txt)
    # Junta e reutiliza o mesmo parser do modo padrão
    joined = "\n".join(linhas_texto)
    itens = split_blocos_por_item(joined)
    registros = [extrai_campos_item_por_regex(b) for b in itens]
    df = pd.DataFrame(registros)
    for col in ["qtd", "valor_unitario", "valor_total"]:
        df[f"{col}_num"] = df[col].apply(to_float_br)
    if "it_code" in df.columns and df["it_code"].notna().any():
        df["_itnum"] = df["it_code"].str.extract(r"(\d+)", expand=False).apply(to_int_safe)
        df = df.sort_values(by=["_itnum", "codigo_produto"], kind="stable").drop(columns=["_itnum"])
    return df


# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="Leitor de Itens da NF-e (DANFE)", layout="wide")
st.title("🧾 Extrator de Itens da Nota Fiscal (DANFE)")

st.markdown(
    """
Este app lê o PDF da **DANFE** e retorna a tabela de **itens** (código, descrição, NCM, CFOP, UN, QTD, V. Unitário, V. Total).
"""
)

arquivo = st.file_uploader("Selecione o PDF da DANFE", type=["pdf"])
modo = st.radio(
    "Modo de extração",
    options=["Padrão (recomendado)", "Avançado (coordenadas)"],
    help="Use o Avançado se o Padrão não capturar algum campo corretamente."
)
mostrar_debug = st.checkbox("Mostrar debug (blocos de texto dos itens)", value=False)

col_dl1, col_dl2 = st.columns(2)

if arquivo is not None:
    file_bytes = arquivo.read()

    with st.spinner("Lendo PDF..."):
        if modo.startswith("Padrão"):
            pages_text = pdf_to_pages_text(file_bytes)
            df = extrair_itens_modo_padrao(pages_text)
            if mostrar_debug:
                # mostra o bloco bruto de itens
                debug_txt = recorta_tabela_itens_por_texto(pages_text)
                st.expander("Ver bloco bruto (Padrão)").write(debug_txt)
        else:
            pages_words = pdf_to_pages_words(file_bytes)
            df = extrair_itens_modo_avancado(pages_words)
            if mostrar_debug:
                st.info("No modo Avançado o debug mostra as linhas reconstruídas.")
                debug_lines = []
                for words in pages_words:
                    for ln in agrupa_linhas_por_y(words):
                        txt = " ".join([w[4] for w in ln])
                        if RE_NCM.search(txt) or RE_CFOP.search(txt):
                            debug_lines.append(txt)
                st.expander("Ver linhas candidatas (Avançado)").write("\n".join(debug_lines))

    if df.empty or df["codigo_produto"].isna().all():
        st.warning("Não foi possível identificar itens. Tente o modo Avançado ou envie um exemplo.")
    else:
        st.success(f"Itens encontrados: {len(df)}")
        # Exibe tabela amigável
        cols_show = [
            "it_code", "codigo_produto", "nm_code", "descricao",
            "ncm", "cfop", "un", "qtd", "valor_unitario", "valor_total"
        ]
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True)

        # Sumário financeiro (quando possível)
        if "valor_total_num" in df.columns and df["valor_total_num"].notna().any():
            st.metric("Soma dos V. Totais", f"R$ {df['valor_total_num'].sum():,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        # Downloads
        with col_dl1:
            csv_bytes = df[cols_show].to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Baixar CSV", data=csv_bytes, file_name="itens_nfe.csv", mime="text/csv")
        with col_dl2:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df[cols_show].to_excel(writer, index=False, sheet_name="Itens")
            out.seek(0)
            st.download_button("📥 Baixar Excel", data=out, file_name="itens_nfe.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Dica: se a DANFE mudar de layout, me mande um exemplo que eu ajusto rapidamente o parser.")
