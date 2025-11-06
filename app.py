# -*- coding: utf-8 -*-
import io
import re
import unicodedata
from typing import List, Tuple, Dict, Optional

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF


# ===============================
# Utils
# ===============================
def no_accents_upper(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return s.upper()


def words_from_pdf(file_bytes: bytes) -> List[List[Tuple[float, float, float, float, str]]]:
    """
    Retorna lista por página. Cada item é (x0, y0, x1, y1, texto).
    """
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            ws = page.get_text("words")  # (x0, y0, x1, y1, "text", block, line, word_no)
            ws = [(w[0], w[1], w[2], w[3], w[4]) for w in ws]
            ws.sort(key=lambda t: (round(t[1], 1), t[0]))
            pages.append(ws)
    return pages


def group_lines(words: List[Tuple[float, float, float, float, str]], tol_y: float = 2.0):
    """
    Agrupa palavras por linha (y). Retorna lista de linhas; cada linha é lista de palavras ordenadas por x.
    """
    linhas = []
    cur_y = None
    cur = []
    for (x0, y0, x1, y1, t) in words:
        if cur_y is None:
            cur_y = y0
            cur = [(x0, y0, x1, y1, t)]
            continue
        if abs(y0 - cur_y) <= tol_y:
            cur.append((x0, y0, x1, y1, t))
        else:
            linhas.append(sorted(cur, key=lambda z: z[0]))
            cur = [(x0, y0, x1, y1, t)]
            cur_y = y0
    if cur:
        linhas.append(sorted(cur, key=lambda z: z[0]))
    return linhas


# ===============================
# Detecção de cabeçalho e colunas
# ===============================
END_MARKERS = [
    "DADOS ADICIONAIS",
    "INFORMACOES COMPLEMENTARES",
    "INFORMAÇÕES COMPLEMENTARES",
    "DADOS ADICIONAIS (COMPLEMENTO)",
    "INFORMACOES ADICIONAIS",
    "INFORMAÇÕES ADICIONAIS",
]


def find_header_positions(lines: List[List[Tuple[float, float, float, float, str]]]) -> Optional[Dict[str, float]]:
    """
    Tenta localizar, numa única linha, os títulos das colunas e retorna o x inicial de cada uma.
    Retorna também a y (linha do cabeçalho) via "__y__".
    Colunas alvo: COD | DESCRICAO | NCM/SH | CFOP | UN | QTD | V_UNITARIO | V_TOTAL
    """
    for ln in lines:
        tokens = [no_accents_upper(w[4]) for w in ln]
        # Exigimos NCM/SH e CFOP na mesma linha (padrão de DANFE)
        if not any("NCM/SH" in t for t in tokens):
            continue
        if "CFOP" not in tokens:
            continue

        col_x = {}
        for i, (x0, y0, x1, y1, text) in enumerate(ln):
            t = no_accents_upper(text)

            # COD PROD: "CÓD." + "PROD."
            if t in {"COD.", "COD", "CÓD.", "CÓD"}:
                nxt = no_accents_upper(ln[i + 1][4]) if i + 1 < len(ln) else ""
                if nxt.startswith("PROD"):
                    col_x["COD"] = x0

            # DESCRIÇÃO
            if t.startswith("DESCRICAO") or t.startswith("DESCRIÇÃO"):
                col_x["DESCRICAO"] = x0

            # NCM/SH
            if "NCM/SH" in t:
                col_x["NCM/SH"] = x0

            # CFOP
            if t == "CFOP":
                col_x["CFOP"] = x0

            # UN
            if t == "UN":
                col_x["UN"] = x0

            # QTD
            if t == "QTD":
                col_x["QTD"] = x0

            # V. UNITÁRIO / V. TOTAL (pode vir como "V." + "UNITÁRIO"/"TOTAL" ou "VALOR UNITÁRIO"/"VALOR TOTAL")
            if t in {"V.", "V"} and i + 1 < len(ln):
                next_t = no_accents_upper(ln[i + 1][4])
                if next_t.startswith("UNIT"):
                    col_x["V_UNITARIO"] = x0
                if next_t.startswith("TOTAL"):
                    col_x["V_TOTAL"] = x0
            if t == "VALOR" and i + 1 < len(ln):
                next_t = no_accents_upper(ln[i + 1][4])
                if next_t.startswith("UNIT"):
                    col_x["V_UNITARIO"] = x0
                if next_t.startswith("TOTAL"):
                    col_x["V_TOTAL"] = x0

        needed = {"COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"}
        have = set(col_x.keys())
        # Considera header válido se achou pelo menos 6 das 8
        if len(needed.intersection(have)) >= 6:
            col_x["__y__"] = ln[0][1]
            return col_x

    return None


def build_column_edges(col_x: Dict[str, float], page_width: float) -> List[Tuple[str, float, float]]:
    """
    Com os x dos títulos, gera intervalos [x_ini, x_fim) para cada coluna.
    """
    keys = [k for k in ["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"] if k in col_x]
    keys.sort(key=lambda k: col_x[k])
    xs = [col_x[k] for k in keys]
    edges = []
    for i, k in enumerate(keys):
        left = (xs[i - 1] + xs[i]) / 2 if i > 0 else max(0, xs[i] - 5)
        right = (xs[i] + xs[i + 1]) / 2 if i + 1 < len(xs) else page_width
        edges.append((k, left, right))
    return edges


# ===============================
# Extração de linhas da tabela
# ===============================
def extract_table_page(page, unite_wrapped_description: bool = True) -> List[Dict[str, str]]:
    """
    Extrai as linhas de uma página, respeitando o cabeçalho encontrado.
    Retorna lista de dicts (linha crua, sem normalização).
    """
    words = page.get_text("words")
    words = [(w[0], w[1], w[2], w[3], w[4]) for w in words]
    words.sort(key=lambda t: (round(t[1], 1), t[0]))

    lines = group_lines(words, tol_y=2.0)
    if not lines:
        return []

    col_x = find_header_positions(lines)
    if not col_x:
        return []

    header_y = col_x["__y__"]
    page_width = page.rect.width
    col_edges = build_column_edges(col_x, page_width)

    rows = []
    past_header = False
    for ln in lines:
        y = ln[0][1]
        row_text_all = " ".join(no_accents_upper(w[4]) for w in ln)

        if not past_header:
            if y <= header_y + 0.5:
                continue
            past_header = True

        if any(m in row_text_all for m in END_MARKERS):
            break

        row = {k: "" for k in ["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]}
        for (x0, y0, x1, y1, t) in ln:
            xc = (x0 + x1) / 2
            placed = False
            for (k, left, right) in col_edges:
                if left <= xc < right:
                    row[k] = (row.get(k, "") + " " + t).strip()
                    placed = True
                    break
            if not placed:
                row["DESCRICAO"] = (row.get("DESCRICAO", "") + " " + t).strip()

        rows.append(row)

    if unite_wrapped_description and rows:
        merged = []
        for r in rows:
            is_continuation = (
                r["COD"].strip() == "" and
                r["NCM/SH"].strip() == "" and
                r["CFOP"].strip() == "" and
                r["UN"].strip() == "" and
                r["QTD"].strip() == "" and
                r["V_UNITARIO"].strip() == "" and
                r["V_TOTAL"].strip() == "" and
                r["DESCRICAO"].strip() != ""
            )
            if is_continuation and merged:
                merged[-1]["DESCRICAO"] = (merged[-1]["DESCRICAO"] + " " + r["DESCRICAO"]).strip()
            else:
                merged.append(r)
        rows = merged

    return rows


def extract_table_full(file_bytes: bytes, unite_wrapped_description: bool = True) -> pd.DataFrame:
    """
    Percorre todas as páginas; em cada uma, detecta cabeçalho e extrai linhas.
    """
    out_rows = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            page_rows = extract_table_page(p, unite_wrapped_description=unite_wrapped_description)
            out_rows.extend(page_rows)

    df = pd.DataFrame(out_rows, columns=["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"])
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # Remove linhas totalmente vazias
    df = df[~(df == "").all(axis=1)].reset_index(drop=True)
    return df


# ===============================
# Organização / Normalização (opcional)
# ===============================
RE_IT = re.compile(r"\bIT\s*\d{2,}\b", re.IGNORECASE)
RE_NM = re.compile(r"\bNM\d{5,}\b", re.IGNORECASE)

def extract_it_nm(descricao: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extrai IT e NM do início da descrição quando existirem.
    Retorna (IT, NM, descricao_sem_prefixos)
    """
    if not isinstance(descricao, str):
        return None, None, descricao
    it = None
    nm = None
    # captura primeira ocorrência
    m_it = RE_IT.search(descricao)
    m_nm = RE_NM.search(descricao)
    # limpa prefixos ' - ' comuns
    clean = descricao
    if m_it:
        it = m_it.group(0).upper()
    if m_nm:
        nm = m_nm.group(0).upper()
    # Remove ambos do início se estiverem próximos
    # Ex.: "IT180 - NM12773524 - VERTEBRA ..." -> "VERTEBRA ..."
    lead = descricao
    lead_upper = no_accents_upper(lead)
    # corta até depois do primeiro " - " após NM/IT
    first_sep = lead.find(" - ")
    if first_sep != -1 and (m_it or m_nm) and (m_it and m_it.start() < 15 or m_nm and m_nm.start() < 15):
        # tenta cortar até depois do segundo " - ", que é onde costuma iniciar a descrição
        second_sep = lead.find(" - ", first_sep + 3)
        if second_sep != -1:
            clean = lead[second_sep + 3 :].strip()
    return it, nm, clean


def to_float_br(s: str) -> Optional[float]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s == "":
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def organize_table(
    df_raw: pd.DataFrame,
    extract_it_nm_flag: bool = True,
    normalize_numbers_flag: bool = True
) -> pd.DataFrame:
    """
    A partir da tabela crua, gera colunas organizadas (opcionalmente IT, NM e numéricos).
    Não remove nenhuma coluna ainda; só adiciona.
    """
    df = df_raw.copy()

    if extract_it_nm_flag and "DESCRICAO" in df.columns:
        its, nms, descrs = [], [], []
        for x in df["DESCRICAO"].fillna(""):
            it, nm, dclean = extract_it_nm(x)
            its.append(it)
            nms.append(nm)
            descrs.append(dclean)
        df["IT"] = its
        df["NM"] = nms
        df["DESCRICAO_LIMPA"] = descrs

    if normalize_numbers_flag:
        for col in ["QTD", "V_UNITARIO", "V_TOTAL"]:
            df[col + "_NUM"] = df[col].apply(to_float_br)

    # Ordena por IT numérico quando existir
    if "IT" in df.columns:
        df["_itnum"] = df["IT"].str.extract(r"(\d+)", expand=False)
        df["_itnum"] = pd.to_numeric(df["_itnum"], errors="coerce")
        df = df.sort_values(by=["_itnum"]).drop(columns=["_itnum"])

    return df


# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="NF-e: Leitura exata + Organização de colunas", layout="wide")
st.title("🧾 NF-e (DANFE) — Leitura EXATA da tabela + Organização")

st.markdown(
    """
1) **Lê a tabela exatamente como está** no PDF (pelo cabeçalho).  
2) **Organiza** e deixa você **escolher quais colunas quer retornar** (e como renomear).  
3) (Opcional) **Extrai IT/NM** e **normaliza números** (QTD / V. Unitário / V. Total).
"""
)

file = st.file_uploader("Selecione o PDF da DANFE", type=["pdf"])
unir_quebras = st.checkbox("Unir linhas quebradas de descrição", value=True)
extract_it_nm_flag = st.checkbox("Extrair IT/NM da descrição", value=True)
normalize_numbers_flag = st.checkbox("Normalizar QTD / V. Unitário / V. Total (pt-BR → número)", value=True)

default_out_cols = [
    "IT", "COD", "NM", "DESCRICAO_LIMPA",
    "NCM/SH", "CFOP", "UN",
    "QTD", "V_UNITARIO", "V_TOTAL",
    "QTD_NUM", "V_UNITARIO_NUM", "V_TOTAL_NUM"
]

st.markdown("#### 3) Seleção das colunas de saída")
selected_cols = st.multiselect(
    "Escolha as colunas que deseja no resultado",
    options=[
        "IT", "NM", "COD", "DESCRICAO", "DESCRICAO_LIMPA",
        "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL",
        "QTD_NUM", "V_UNITARIO_NUM", "V_TOTAL_NUM"
    ],
    default=default_out_cols
)

st.markdown("#### 4) Renomear colunas (opcional)")
rename_map = {}
if selected_cols:
    cols1, cols2 = st.columns(2)
    half = (len(selected_cols) + 1) // 2
    with cols1:
        for c in selected_cols[:half]:
            new = st.text_input(f"Renomear '{c}' para:", value=c)
            if new and new != c:
                rename_map[c] = new
    with cols2:
        for c in selected_cols[half:]:
            new = st.text_input(f"Renomear '{c}' para:", value=c)
            if new and new != c:
                rename_map[c] = new

colA, colB = st.columns([1, 1])
with colA:
    btn_extract = st.button("📤 Extrair e Organizar")
with colB:
    auto_download = st.checkbox("Mostrar botões de download", value=True)

if btn_extract and file is not None:
    raw = file.read()
    with st.spinner("Lendo tabela diretamente do PDF..."):
        df_raw = extract_table_full(raw, unite_wrapped_description=unir_quebras)

    if df_raw.empty:
        st.error("Não localizei a tabela de itens nesta DANFE. Me envie o PDF para calibrar o cabeçalho.")
    else:
        st.success(f"Tabela crua capturada — linhas: {len(df_raw)}")
        st.expander("Visualizar tabela CRUA (como está no PDF)").dataframe(df_raw, use_container_width=True, height=300)

        with st.spinner("Organizando colunas..."):
            df_org = organize_table(
                df_raw,
                extract_it_nm_flag=extract_it_nm_flag,
                normalize_numbers_flag=normalize_numbers_flag
            )

            # Seleciona e renomeia
            cols_final = [c for c in selected_cols if c in df_org.columns]
            if not cols_final:
                st.warning("Nenhuma coluna selecionada existe no dataframe. Verifique a seleção.")
            else:
                df_out = df_org[cols_final].rename(columns=rename_map)

                st.subheader("Resultado final")
                st.dataframe(df_out, use_container_width=True, height=420)

                # Métrica de soma do total, se disponível
                total_col = "V_TOTAL_NUM" if "V_TOTAL_NUM" in df_org.columns and "V_TOTAL_NUM" in cols_final else None
                if total_col:
                    soma_total = df_out[rename_map.get(total_col, total_col)].sum(skipna=True)
                    st.metric("Soma do V. Total (numérico)", f"R$ {soma_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

                if auto_download:
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("📥 Baixar CSV (organizado)", data=csv_bytes, file_name="itens_nfe_organizado.csv", mime="text/csv")

                    xls = io.BytesIO()
                    with pd.ExcelWriter(xls, engine="openpyxl") as w:
                        df_out.to_excel(w, index=False, sheet_name="Itens")
                    xls.seek(0)
                    st.download_button(
                        "📥 Baixar Excel (organizado)",
                        data=xls,
                        file_name="itens_nfe_organizado.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

st.markdown("---")
st.caption("Caso você já saiba as colunas finais exatas, me diga quais são e eu fixo no código para sair direto do jeito que você precisa.")
