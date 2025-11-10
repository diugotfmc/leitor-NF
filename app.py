# -*- coding: utf-8 -*-
import io
import re
import unicodedata
from typing import List, Tuple, Dict, Optional

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF


# ===============================
# Utilitários básicos
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


# ===============================
# Detecção do cabeçalho + colunas
# ===============================
def find_header_positions(lines: List[List[Tuple[float, float, float, float, str]]]) -> Optional[Dict[str, float]]:
    """
    Localiza o cabeçalho da tabela e retorna o x de cada coluna + y do header em "__y__".
    Alvo primário: COD | DESCRICAO | NCM/SH | (CST?) | CFOP | UN | QTD | V_UNITARIO | V_TOTAL
    O 'CST' pode ou não aparecer como coluna; se não vier, trataremos por sanitização.
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

            # (Opcional) CST — algumas DANFEs trazem coluna "CST"
            if t == "CST":
                col_x["CST"] = x0

            # CFOP
            if t == "CFOP":
                col_x["CFOP"] = x0

            # UN
            if t == "UN":
                col_x["UN"] = x0

            # QTD
            if t == "QTD":
                col_x["QTD"] = x0

            # V. UNITÁRIO / V. TOTAL (ou "VALOR UNITÁRIO"/"VALOR TOTAL")
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
        need = {"COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD"}  # CST é desejável, mas não obrigatório
        if len(have.intersection(need)) >= 6:
            col_x["__y__"] = ln[0][1]
            return col_x
    return None


def build_column_edges(col_x: Dict[str, float], page_width: float) -> List[Tuple[str, float, float]]:
    """Gera intervalos [x_ini, x_fim) para as colunas encontradas no header."""
    keys_pref = ["COD", "DESCRICAO", "NCM/SH", "CST", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]
    keys = [k for k in keys_pref if k in col_x]
    keys.sort(key=lambda k: col_x[k])
    xs = [col_x[k] for k in keys]
    edges = []
    for i, k in enumerate(keys):
        left = (xs[i - 1] + xs[i]) / 2 if i > 0 else max(0, xs[i] - 5)
        right = (xs[i] + xs[i + 1]) / 2 if i + 1 < len(xs) else page_width
        edges.append((k, left, right))
    return edges


# ===============================
# Extração por página (linhas cruas)
# ===============================
def _append_text(base: str, extra: str) -> str:
    """Une texto cuidando de hifenização e espaços."""
    base = base.rstrip()
    extra = extra.lstrip()
    if base.endswith("-"):
        return base[:-1] + extra
    if base:
        return base + " " + extra
    return extra


def extract_table_page(page) -> List[Dict[str, str]]:
    """
    Mapeia palavras -> colunas via cabeçalho desta página.
    Retorna 'linhas cruas' (cada linha visual, sem consolidação por item ainda).
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

        row = {k: "" for k in ["COD", "DESCRICAO", "NCM/SH", "CST", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]}
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

        # limpeza leve
        for k in row:
            row[k] = " ".join(str(row[k]).replace("\n", " ").split())

        raw_rows.append(row)

    return raw_rows


# ===============================
# Consolidação (linha única por item) + Sanitização NCM/CST/CFOP
# ===============================
def is_new_item(row: Dict[str, str]) -> bool:
    """Novo item quando a linha traz COD e NCM/SH visivelmente preenchidos."""
    return bool(row.get("COD", "").strip()) and bool(row.get("NCM/SH", "").strip())


def parse_ncm_cst_cfop(ncm_text: str, cst_text: str, cfop_text: str) -> Tuple[str, str, str]:
    """
    Recebe textos crus das 3 colunas e devolve (NCM8, CST3, CFOP4) limpos.
    Estratégia:
     1) Tokeniza dígitos: do NCM text (e, se preciso, concatena com cst/cfop text).
     2) Primeiro 8 dígitos -> NCM
     3) Depois do NCM, próximo 3 dígitos -> CST
     4) Depois, próximo 4 dígitos -> CFOP
    Se alguma etapa não achar, usa fallback do campo correspondente (busca direta).
    """
    def tokens_digits(s: str) -> List[str]:
        return [t for t in re.split(r"\D+", s or "") if t]

    toks = tokens_digits((ncm_text or "") + " " + (cst_text or "") + " " + (cfop_text or ""))

    ncm = cst = cfop = ""
    i = 0
    # NCM
    while i < len(toks):
        if len(toks[i]) == 8:
            ncm = toks[i]
            i += 1
            break
        i += 1
    # CST
    while i < len(toks):
        if len(toks[i]) == 3:
            cst = toks[i]
            i += 1
            break
        i += 1
    # CFOP
    while i < len(toks):
        if len(toks[i]) == 4:
            cfop = toks[i]
            i += 1
            break
        i += 1

    # Fallbacks por coluna individual (caso venham separadas)
    if not ncm:
        m = re.search(r"\b(\d{8})\b", ncm_text or "")
        if m: ncm = m.group(1)
    if not cst:
        m = re.search(r"\b(\d{3})\b", cst_text or "")
        if m: cst = m.group(1)
    if not cfop:
        m = re.search(r"\b(\d{4})\b", cfop_text or "")
        if m: cfop = m.group(1)

    return ncm, cst, cfop


def consolidate_rows_into_items(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Consolida N linhas em 1 item (descrição unificada).
    - Inicia novo item com (COD + NCM/SH).
    - Linhas seguintes agregam DESCRICAO até o próximo item.
    - Sanitiza NCM/SH (8 dígitos), CST (3 dígitos) e CFOP (4 dígitos).
    """
    final_rows = []
    current = None

    for r in raw_rows:
        if is_new_item(r):
            if current:
                # Saneia tributos ao fechar item
                ncm, cst, cfop = parse_ncm_cst_cfop(current.get("NCM/SH", ""), current.get("CST", ""), current.get("CFOP", ""))
                current["NCM/SH"], current["CST"], current["CFOP"] = ncm, cst, cfop
                final_rows.append(current)

            current = {k: r.get(k, "").strip() for k in ["COD", "DESCRICAO", "NCM/SH", "CST", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"]}
            continue

        # Continuação
        if current:
            if r.get("DESCRICAO", "").strip():
                current["DESCRICAO"] = _append_text(current["DESCRICAO"], r["DESCRICAO"])
            # completa campos estruturais apenas se ainda vazios no item
            for col in ["CST", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL", "NCM/SH"]:
                if not current.get(col, "").strip() and r.get(col, "").strip():
                    current[col] = r[col].strip()
        else:
            continue

    if current:
        ncm, cst, cfop = parse_ncm_cst_cfop(current.get("NCM/SH", ""), current.get("CST", ""), current.get("CFOP", ""))
        current["NCM/SH"], current["CST"], current["CFOP"] = ncm, cst, cfop
        final_rows.append(current)

    # limpeza final
    for r in final_rows:
        for k in r:
            r[k] = " ".join(str(r[k]).split())

    return final_rows


def extract_table_full(file_bytes: bytes) -> pd.DataFrame:
    """Extrai de todas as páginas e devolve uma linha por item (NCM/SH, CST, CFOP saneados)."""
    out_rows = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            raw_rows = extract_table_page(p)
            out_rows.extend(raw_rows)

    items = consolidate_rows_into_items(out_rows)
    df = pd.DataFrame(items, columns=["COD", "DESCRICAO", "NCM/SH", "CST", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"])
    return df


# ===============================
# Organização nas 6 colunas pedidas
# (mantidas; CFOP/CST aparecem na tabela consolidada)
# ===============================
RE_IT = re.compile(r"\bIT\s*\d{2,}\b", re.IGNORECASE)
RE_NM_NUM = re.compile(r"\bNM\.?\s*(\d{5,})\b", re.IGNORECASE)

def extrair_item_unifilar(descricao: str) -> str:
    m = RE_IT.search(descricao or "")
    return m.group(0).upper().replace(" ", "") if m else ""

def extrair_nm(descricao: str) -> str:
    m = RE_NM_NUM.search(descricao or "")
    return m.group(1) if m else ""

def extrair_descricao_pos_nm(descricao: str) -> str:
    """
    Retorna o 1º segmento descritivo após 'NM...' (quando houver).
    Se não houver 'NM', retorna a descrição consolidada inteira.
    """
    if not isinstance(descricao, str):
        return ""
    parts = [p.strip() for p in re.split(r"\s*-\s*", descricao)]
    idx_nm = None
    for i, p in enumerate(parts):
        if RE_NM_NUM.search(p):
            idx_nm = i
            break
    if idx_nm is None:
        return " ".join(parts).strip()
    if idx_nm + 1 < len(parts):
        return parts[idx_nm + 1].strip()
    return " ".join(parts).strip()


def organizar_para_seis_colunas(df_itens: pd.DataFrame) -> pd.DataFrame:
    """
    Desenho, Item Unifilar, NM, Descrição, QTD, Unidade.
    """
    df = pd.DataFrame(columns=["Desenho", "Item Unifilar", "NM", "Descrição", "QTD", "Unidade"])
    if df_itens.empty:
        return df

    desen = df_itens["COD"].fillna("").astype(str)
    desc  = df_itens["DESCRICAO"].fillna("").astype(str)
    qtd   = df_itens["QTD"].fillna("").astype(str)
    un    = df_itens["UN"].fillna("").astype(str)

    df["Desenho"] = desen
    df["Item Unifilar"] = desc.apply(extrair_item_unifilar)
    df["NM"] = desc.apply(extrair_nm)
    df["Descrição"] = desc.apply(extrair_descricao_pos_nm)
    df["QTD"] = qtd
    df["Unidade"] = un

    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df


# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="NF-e (DANFE) — Itens consolidados + NCM/SH/CFOP/CST saneados", layout="wide")
st.title("🧾 NF-e (DANFE) — Linha única por item + NCM/SH / CFOP / CST garantidos")

st.markdown(
    """
Este app:
- Lê a tabela pelo **cabeçalho** e consolida **várias linhas** em **UMA linha por item** (descrição não é quebrada);
- Garante que:
  - **NCM/SH** = primeiro código de **8 dígitos** (ex.: `73269090`);
  - **CST** = primeiro código de **3 dígitos** após o NCM (ex.: `000`);
  - **CFOP** = primeiro código de **4 dígitos** encontrado após CST (ex.: `5102`).
- Mostra também a visão com as 6 colunas: **Desenho, Item Unifilar, NM, Descrição, QTD, Unidade**.
"""
)

file = st.file_uploader("Selecione o PDF da DANFE", type=["pdf"])
btn = st.button("📤 Extrair")

if btn and file is not None:
    raw = file.read()
    with st.spinner("Lendo a tabela e consolidando itens..."):
        df_items = extract_table_full(raw)

    if df_items.empty:
        st.error("Não consegui localizar/consolidar a tabela de itens. Se puder, me envie o PDF para calibrarmos.")
    else:
        st.success(f"Itens consolidados (uma linha por item): {len(df_items)}")

        st.subheader("1) Itens consolidados (NCM/SH, CST, CFOP saneados)")
        st.dataframe(df_items, use_container_width=True, height=420)

        # export consolidados
        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = df_items.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Baixar CSV (consolidados)", data=csv_bytes, file_name="itens_consolidados.csv", mime="text/csv")
        with c2:
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

        st.subheader("2) Visão organizada (6 colunas)")
        df6 = organizar_para_seis_colunas(df_items)
        st.dataframe(df6, use_container_width=True, height=420)

        # export 6 colunas
        c3, c4 = st.columns(2)
        with c3:
            csv6 = df6.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Baixar CSV (6 colunas)", data=csv6, file_name="itens_6_colunas.csv", mime="text/csv")
        with c4:
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
