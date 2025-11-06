# -*- coding: utf-8 -*-
import io
import unicodedata
import re
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
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            ws = page.get_text("words")  # (x0, y0, x1, y1, "text", block, line, word_no)
            ws = [(w[0], w[1], w[2], w[3], w[4]) for w in ws]
            ws.sort(key=lambda t: (round(t[1], 1), t[0]))
            pages.append(ws)
    return pages


def group_lines(words: List[Tuple[float, float, float, float, str]], tol_y: float = 2.0):
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
    Localiza a linha de cabeçalho da tabela (CÓD. PROD., DESCRIÇÃO, NCM/SH, CFOP, UN, QTD, V. UNITÁRIO, V. TOTAL)
    e retorna o x de cada coluna + y do header em "__y__".
    """
    for ln in lines:
        tokens = [no_accents_upper(w[4]) for w in ln]
        # precisa conter NCM/SH e CFOP
        if not any("NCM/SH" in t for t in tokens):
            continue
        if not any("CFOP" in t for t in tokens):
            continue

        col_x = {}
        for i, (x0, y0, x1, y1, text) in enumerate(ln):
            t = no_accents_upper(text)

            # COD PROD
            if t in {"COD.", "COD", "CÓD.", "CÓD"}:
                nxt = no_accents_upper(ln[i + 1][4]) if i + 1 < len(ln) else ""
                if nxt.startswith("PROD"):
                    col_x["COD"] = x0

            # DESCRICAO
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

            # V. UNITARIO / V. TOTAL (ou VALOR UNITARIO / VALOR TOTAL)
            if t in {"V.", "V", "VALOR"} and i + 1 < len(ln):
                next_t = no_accents_upper(ln[i + 1][4])
                if next_t.startswith("UNIT"):
                    col_x["V_UNITARIO"] = x0
                if next_t.startswith("TOTAL"):
                    col_x["V_TOTAL"] = x0

        needed = {"COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD"}
        if len(needed.intersection(set(col_x.keys()))) >= 6:
            col_x["__y__"] = ln[0][1]
            return col_x
    return None


def build_column_edges(col_x: Dict[str, float], page_width: float) -> List[Tuple[str, float, float]]:
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
# Extração da tabela (crua)
# ===============================
def extract_table_page(page, unite_wrapped_description: bool = True) -> List[Dict[str, str]]:
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
                r["COD"].strip() == "" and r["NCM/SH"].strip() == "" and r["CFOP"].strip() == "" and
                r["UN"].strip() == "" and r["QTD"].strip() == "" and r["V_UNITARIO"].strip() == "" and
                r["V_TOTAL"].strip() == "" and r["DESCRICAO"].strip() != ""
            )
            if is_continuation and merged:
                merged[-1]["DESCRICAO"] = (merged[-1]["DESCRICAO"] + " " + r["DESCRICAO"]).strip()
            else:
                merged.append(r)
        rows = merged

    return rows


def extract_table_full(file_bytes: bytes, unite_wrapped_description: bool = True) -> pd.DataFrame:
    out_rows = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            page_rows = extract_table_page(p, unite_wrapped_description=unite_wrapped_description)
            out_rows.extend(page_rows)

    if not out_rows:
        return pd.DataFrame(columns=["COD", "DESCRICAO", "NCM/SH", "CFOP", "UN", "QTD", "V_UNITARIO", "V_TOTAL"])

    df = pd.DataFrame(out_rows)
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df


# ===============================
# Organização nas 6 colunas pedidas
# ===============================
RE_IT = re.compile(r"\bIT\s*\d{2,4}\b", flags=re.IGNORECASE)
RE_NM = re.compile(r"\bNM\.?\s*(\d{5,})\b", flags=re.IGNORECASE)

def extrair_item_unifilar(texto: str) -> Optional[str]:
    m = RE_IT.search(texto or "")
    if m:
        return m.group(0).upper().replace(" ", "")
    return None

def extrair_nm(texto: str) -> Optional[str]:
    m = RE_NM.search(texto or "")
    if m:
        return m.group(1)  # só os dígitos
    return None

def extrair_descricao_principal(texto: str) -> Optional[str]:
    """
    A descrição principal é o primeiro segmento léxico após o 'NM...' na linha,
    ou, em fallback, o primeiro segmento que não seja IT/NM/BJ/TR/POS/AC... contendo letras.
    """
    if not texto:
        return None
    # Normalizamos somente para checagens; retornamos no texto original (sem perder acentos)
    tokens = [t.strip() for t in re.split(r"\s*[-–]\s*", texto)]
    # 1) após NM...
    seen_nm = False
    for t in tokens:
        if RE_NM.search(t):
            seen_nm = True
            continue
        if seen_nm:
            return t.strip(" -—–") or None
    # 2) fallback: primeiro bloco descritivo
    for t in tokens:
        up = no_accents_upper(t)
        if any(up.startswith(pfx) for pfx in ["IT", "NM", "BJ", "TR", "POS", "AC0", "AC1", "AC2", "AC3", "AC4", "AC5", "AC6", "AC7", "AC8", "AC9"]):
            continue
        if re.search(r"[A-Za-zÀ-ÿ]", t):  # tem letras
            return t.strip(" -—–")
    return texto.strip()

def organizar_tabela(df_cru: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia a tabela crua (como lida do PDF) para:
    Desenho, Item Unifilar, NM, Descrição, QTD, Unidade.
    """
    if df_cru.empty:
        return pd.DataFrame(columns=["Desenho", "Item Unifilar", "NM", "Descrição", "QTD", "Unidade"])

    def _row_map(r):
        desenho = (r.get("COD") or "").strip()
        descraw = (r.get("DESCRICAO") or "").strip()
        it = extrair_item_unifilar(descraw) or ""
        nm = extrair_nm(descraw) or ""
        descricao = extrair_descricao_principal(descraw) or ""
        qtd = (r.get("QTD") or "").strip()
        un = (r.get("UN") or "").strip()
        return pd.Series({
            "Desenho": desenho,
            "Item Unifilar": it,
            "NM": nm,
            "Descrição": descricao,
            "QTD": qtd,
            "Unidade": un
        })

    df_out = df_cru.apply(_row_map, axis=1)

    # limpeza final de espaços múltiplos
    for c in df_out.columns:
        df_out[c] = df_out[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    return df_out


# ===============================
# UI Streamlit
# ===============================
st.set_page_config(page_title="Leitura e Organização da Tabela da NF (DANFE)", layout="wide")
st.title("🧾 NF-e (DANFE) → Tabela Crua + Tabela Organizada")

st.markdown(
    """
1) **Lemos a tabela exatamente como está no PDF** (sem heurísticas).  
2) **Organizamos** nos campos: **Desenho, Item Unifilar, NM, Descrição, QTD, Unidade**.
"""
)

file = st.file_uploader("Selecione o PDF da DANFE", type=["pdf"])
unir_quebras = st.checkbox("Unir linhas quebradas de descrição (recomendado)", value=True)
btn = st.button("📤 Extrair e Organizar")

if btn and file is not None:
    raw = file.read()

    with st.spinner("Lendo a tabela diretamente do PDF..."):
        df_cru = extract_table_full(raw, unite_wrapped_description=unir_quebras)

    if df_cru.empty:
        st.error("Não consegui localizar o cabeçalho da tabela nesta DANFE ou não havia linhas sob o cabeçalho.")
    else:
        st.success(f"Leitura concluída. Linhas capturadas: {len(df_cru)}")

        tab1, tab2 = st.tabs(["Tabela Crua (como no PDF)", "Tabela Organizada (6 colunas)"])

        with tab1:
            st.dataframe(df_cru, use_container_width=True, height=420)
            csv1 = df_cru.to_csv(index=False).encode("utf-8-sig")
            xls1 = io.BytesIO()
            with pd.ExcelWriter(xls1, engine="openpyxl") as w:
                df_cru.to_excel(w, index=False, sheet_name="Crua")
            xls1.seek(0)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 CSV (crua)", data=csv1, file_name="itens_danfe_cru.csv", mime="text/csv")
            with col2:
                st.download_button(
                    "📥 Excel (crua)",
                    data=xls1,
                    file_name="itens_danfe_cru.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with tab2:
            df_org = organizar_tabela(df_cru)
            st.dataframe(df_org, use_container_width=True, height=420)
            csv2 = df_org.to_csv(index=False).encode("utf-8-sig")
            xls2 = io.BytesIO()
            with pd.ExcelWriter(xls2, engine="openpyxl") as w:
                df_org.to_excel(w, index=False, sheet_name="Organizada")
            xls2.seek(0)
            col3, col4 = st.columns(2)
            with col3:
                st.download_button("📥 CSV (organizada)", data=csv2, file_name="itens_danfe_organizada.csv", mime="text/csv")
            with col4:
                st.download_button(
                    "📥 Excel (organizada)",
                    data=xls2,
                    file_name="itens_danfe_organizada.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

st.markdown("---")
st.caption("Se algum fornecedor imprimir a DANFE com layout muito diferente, me mande um exemplo para eu adaptar o detector de colunas.")
