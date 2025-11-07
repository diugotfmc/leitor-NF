# -*- coding: utf-8 -*-
import io
import re
import unicodedata
from typing import List, Tuple, Dict, Optional

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF


# ===============================
# Utilitários
# ===============================
def no_accents_upper(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return s.upper()


def group_lines(words: List[Tuple[float, float, float, float, str]], tol_y: float = 2.0):
    """Agrupa palavras por linha (y)."""
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
    Localiza o cabeçalho da tabela e retorna o x de cada coluna + y do header em "__y__".
    Alvo: COD | DESCRICAO | NCM/SH | CFOP | UN | QTD | V_UNITARIO | V_TOTAL
    """
    for ln in lines:
        tokens = [no_accents_upper(w[4]) for w in ln]
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

            # V. UNITÁRIO / V. TOTAL
            if t in {"V.", "V"} and i + 1 < len(ln):
                nxt = no_accents_upper(ln[i + 1][4])
                if nxt.startswith("UNIT"):
                    col_x["V_UNITARIO"] = x0
                if nxt.startswith("TOTAL"):
                    col_x["V_TOTAL"] = x0
            if t == "VALOR" and i + 1 < len(ln):
                nxt = no_accents_upper(ln[i + 1][4])
                if nxt.startswith("UNIT"):
                    col_x["V_UNITARIO"] = x0
                if nxt.startswith("TOTAL"):
                    col_x["V_TOTAL"] = x0

        have = set(col_x.keys())
        need = {"COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD"}
        if len(have.intersection(need)) >= 6:
            col_x["__y__"] = ln[0][1]
            return col_x
    return None


def build_column_edges(col_x: Dict[str, float], page_width: float) -> List[Tuple[str, float, float]]:
    """Gera intervalos [x_ini, x_fim) para cada coluna encontrado no header."""
    keys = [k for k in ["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"] if k in col_x]
    keys.sort(key=lambda k: col_x[k])
    xs = [col_x[k] for k in keys]
    edges = []
    for i, k in enumerate(keys):
        left = (xs[i - 1] + xs[i]) / 2 if i > 0 else max(0, xs[i] - 5)
        right = (xs[i] + xs[i + 1]) / 2 if i + 1 < len(xs) else page_width
        edges.append((k, left, right))
    return edges


def _append_text(base: str, extra: str) -> str:
    """Une texto cuidando de hifenização e espaços."""
    base = base.rstrip()
    extra = extra.lstrip()
    if base.endswith("-"):
        return base[:-1] + extra
    if base:
        return base + " " + extra
    return extra


def extract_table_page(page, unite_wrapped_description: bool = True) -> List[Dict[str, str]]:
    """
    Mapeia palavras -> colunas via cabeçalho desta página.
    Retorna 'linhas cruas' (ainda não consolidadas por item).
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

    raw_rows = []
    past_header = False
    for ln in lines:
        y = ln[0][1]
        row_text_all_upper = " ".join(no_accents_upper(w[4]) for w in ln)

        if not past_header:
            if y <= header_y + 0.5:
                continue
            past_header = True

        if any(m in row_text_all_upper for m in END_MARKERS):
            break

        row = {k: "" for k in ["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]}
        for (x0, y0, x1, y1, t) in ln:
            xc = (x0 + x1) / 2
            placed = False
            for (k, left, right) in col_edges:
                if left <= xc < right:
                    row[k] = _append_text(row.get(k, ""), t)
                    placed = True
                    break
            if not placed:
                row["DESCRICAO"] = _append_text(row.get("DESCRICAO", ""), t)

        raw_rows.append(row)

    # Só limpeza suave (sem normalizar números)
    for r in raw_rows:
        for k in r:
            r[k] = " ".join(str(r[k]).replace("\n", " ").split())

    return raw_rows


def is_new_item(row: Dict[str, str]) -> bool:
    """
    Decide se a 'row' começa um novo item:
    Regra: tem COD e NCM/SH preenchidos (característico da linha principal do item na DANFE).
    """
    return bool(row.get("COD", "").strip()) and bool(row.get("NCM/SH", "").strip())


def consolidate_rows_into_items(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Consolida N linhas da DANFE em 1 item, preservando a descrição 'inteira'.
    - Inicia novo item quando encontra (COD + NCM/SH).
    - Linhas seguintes, até o próximo item, são tratadas como continuação.
    - Em continuação, apenas DESCRICAO é agregada; campos estruturais só preenchem se o atual estiver vazio.
    """
    final_rows = []
    current = None

    for r in raw_rows:
        if is_new_item(r):
            if current:
                final_rows.append(current)
            current = {k: r.get(k, "").strip() for k in ["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]}
            continue

        # Continuação
        if current:
            # agrega apenas a descrição (sem quebrar linha)
            if r.get("DESCRICAO", "").strip():
                current["DESCRICAO"] = _append_text(current["DESCRICAO"], r["DESCRICAO"])
            # se alguma coluna estrutural veio "perdida" e ainda está vazia no item atual, preenche
            for col in ["CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]:
                if not current.get(col, "").strip() and r.get(col, "").strip():
                    current[col] = r[col].strip()
        else:
            # Linha perdida antes do primeiro item (ignora)
            continue

    if current:
        final_rows.append(current)

    # limpeza final de espaços
    for r in final_rows:
        for k in r:
            r[k] = " ".join(str(r[k]).split())
    return final_rows


def extract_table_full(file_bytes: bytes) -> pd.DataFrame:
    """Extrai e consolida itens em todas as páginas."""
    out_rows = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            raw_rows = extract_table_page(p, unite_wrapped_description=True)
            out_rows.extend(raw_rows)

    items = consolidate_rows_into_items(out_rows)
    df = pd.DataFrame(items, columns=["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"])
    return df


# ===============================
# Organização para 6 colunas pedidas
# ===============================
RE_IT = re.compile(r"\bIT\s*\d{2,}\b", re.IGNORECASE)
RE_NM = re.compile(r"\bNM\.?\s*(\d{5,})\b", re.IGNORECASE)

def extrair_item_unifilar(descricao: str) -> str:
    m = RE_IT.search(descricao or "")
    if m:
        return m.group(0).upper().replace(" ", "")
    return ""

def extrair_nm(descricao: str) -> str:
    m = RE_NM.search(descricao or "")
    if m:
        return m.group(1)
    return ""

def extrair_descricao_pos_nm(descricao: str) -> str:
    """
    Retorna o 1º segmento descritivo após 'NM...' (quando houver).
    Se não houver 'NM', retorna a descrição inteira já consolidada.
    """
    if not isinstance(descricao, str):
        return ""
    parts = [p.strip() for p in re.split(r"\s*-\s*", descricao)]
    # procura o índice do trecho que contém NM...
    idx_nm = None
    for i, p in enumerate(parts):
        if RE_NM.search(p):
            idx_nm = i
            break
    if idx_nm is None:
        return " ".join(parts).strip()
    # descrição costuma iniciar após NM -> pegue o próximo bloco
    if idx_nm + 1 < len(parts):
        return parts[idx_nm + 1].strip()
    return " ".join(parts).strip()


def organizar_para_seis_colunas(df_itens: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia para: Desenho, Item Unifilar, NM, Descrição, QTD, Unidade.
    - Desenho  <- COD
    - Item Unifilar <- IT extraído da DESCRICAO
    - NM       <- NM extraído da DESCRICAO
    - Descrição <- trecho após NM... (ou a própria DESCRICAO consolidada)
    - QTD      <- QTD
    - Unidade  <- UN
    """
    df = pd.DataFrame(columns=["Desenho", "Item Unifilar", "NM", "Descrição", "QTD", "Unidade"])
    if df_itens.empty:
        return df

    desen = df_itens["COD"].fillna("").astype(str)
    desc  = df_itens["DESCRICAO"].fillna("").astype(str)
    qtd   = df_itens["QTD"].fillna("").astype(str)
    un    = df_itens["UN"].fillna("").astype(str)

    itcol = desc.apply(extrair_item_unifilar)
    nmcol = desc.apply(extrair_nm)
    desc_final = desc.apply(extrair_descricao_pos_nm)

    df["Desenho"] = desen
    df["Item Unifilar"] = itcol
    df["NM"] = nmcol
    df["Descrição"] = desc_final
    df["QTD"] = qtd
    df["Unidade"] = un

    # limpeza de espaços
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df


# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="NF-e (DANFE) — Itens consolidados (sem quebra por linha)", layout="wide")
st.title("🧾 NF-e (DANFE) — Extração por tabela, com LINHA ÚNICA por item")

st.markdown(
    """
Este app:
- Lê a tabela pelo **cabeçalho da DANFE**;
- **Consolida** linhas quebradas para formar **uma linha por item** (descrição unificada, sem quebras);
- Mostra também a visão **organizada** com as 6 colunas solicitadas: **Desenho, Item Unifilar, NM, Descrição, QTD, Unidade**.
"""
)

file = st.file_uploader("Selecione o PDF da DANFE", type=["pdf"])
btn = st.button("📤 Extrair")

if btn and file is not None:
    raw = file.read()
    with st.spinner("Lendo a tabela e consolidando itens..."):
        df_items = extract_table_full(raw)

    if df_items.empty:
        st.error("Não consegui localizar/consolidar a tabela de itens. Me envie o PDF para calibrarmos.")
    else:
        st.success(f"Itens consolidados (uma linha por item): {len(df_items)}")
        st.subheader("1) Itens consolidados (como na DANFE, sem quebras de linha)")
        st.dataframe(df_items, use_container_width=True, height=420)

        # Export consolidados
        col1, col2 = st.columns(2)
        with col1:
            csv_bytes = df_items.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Baixar CSV (consolidados)", data=csv_bytes, file_name="itens_consolidados.csv", mime="text/csv")
        with col2:
            xls = io.BytesIO()
            with pd.ExcelWriter(xls, engine="openpyxl") as w:
                df_items.to_excel(w, index=False, sheet_name="Consolidados")
            xls.seek(0)
            st.download_button(
                "📥 Baixar Excel (consolidados)",
                data=xls,
                file_name="itens_consolidados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.subheader("2) Visão organizada (6 colunas pedidas)")
        df6 = organizar_para_seis_colunas(df_items)
        st.dataframe(df6, use_container_width=True, height=420)

        # Export 6 colunas
        col3, col4 = st.columns(2)
        with col3:
            csv6 = df6.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Baixar CSV (6 colunas)", data=csv6, file_name="itens_6_colunas.csv", mime="text/csv")
        with col4:
            xls6 = io.BytesIO()
            with pd.ExcelWriter(xls6, engine="openpyxl") as w:
                df6.to_excel(w, index=False, sheet_name="6 colunas")
            xls6.seek(0)
            st.download_button(
                "📥 Baixar Excel (6 colunas)",
                data=xls6,
                file_name="itens_6_colunas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
