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
# Cabeçalho / colunas
# ===============================
def find_header_positions(lines: List[List[Tuple[float, float, float, float, str]]]) -> Optional[Dict[str, float]]:
    """
    Localiza o cabeçalho da tabela e retorna o x de cada coluna + y do header em "__y__".
    Alvo: COD | DESCRICAO | NCM/SH | (CST?) | CFOP | UN | QTD | V_UNITARIO | V_TOTAL
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

            # COD PROD
            if t in {"COD.", "COD", "CÓD.", "CÓD"}:
                nxt = no_accents_upper(ln[i + 1][4]) if i + 1 < len(ln) else ""
                if nxt.startswith("PROD"):
                    col_x["COD"] = x0

            if t.startswith("DESCRICAO") or t.startswith("DESCRIÇÃO"):
                col_x["DESCRICAO"] = x0

            if "NCM/SH" in t:
                col_x["NCM/SH"] = x0

            if t == "CST":
                col_x["CST"] = x0

            if t == "CFOP":
                col_x["CFOP"] = x0

            if t == "UN":
                col_x["UN"] = x0

            if t == "QTD":
                col_x["QTD"] = x0

            # V. UNITÁRIO / V. TOTAL (ou VALOR ...)
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
# Consolidação + sanitização NCM/CST/CFOP
# ===============================
def is_new_item(row: Dict[str, str]) -> bool:
    """Novo item quando a linha traz COD e NCM/SH visivelmente preenchidos."""
    return bool(row.get("COD", "").strip()) and bool(row.get("NCM/SH", "").strip())


def parse_ncm_cst_cfop(ncm_text: str, cst_text: str, cfop_text: str) -> Tuple[str, str, str]:
    """
    Devolve (NCM8, CST3, CFOP4) limpos a partir dos textos crus.
    Regra: tokeniza dígitos, e pega nessa ordem: 8 -> 3 -> 4. Depois fallbacks por coluna.
    """
    def tokens_digits(s: str) -> List[str]:
        return [t for t in re.split(r"\D+", s or "") if t]

    toks = tokens_digits((ncm_text or "") + " " + (cst_text or "") + " " + (cfop_text or ""))

    ncm = cst = cfop = ""
    i = 0
    while i < len(toks):
        if len(toks[i]) == 8:
            ncm = toks[i]; i += 1; break
        i += 1
    while i < len(toks):
        if len(toks[i]) == 3:
            cst = toks[i]; i += 1; break
        i += 1
    while i < len(toks):
        if len(toks[i]) == 4:
            cfop = toks[i]; i += 1; break
        i += 1

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
# Extração do Nº da NF e da Chave
# ===============================
def extract_access_key_and_nf_number(file_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Retorna (chave_de_acesso_44, numero_nf_9_digitos) a partir do 1º cabeçalho da DANFE.
    1) Procura "CHAVE DE ACESSO" e captura os 44 dígitos subsequentes (permitindo espaços).
    2) A partir da chave, extrai o número da NF (nNF = 9 dígitos).
       Composição oficial da chave (posições 1-based):
         cUF(2) AAMM(4) CNPJ(14) mod(2) série(3) nNF(9) tpEmis(1) cNF(8) cDV(1)
       Ou seja, nNF = dígitos 26..34 (1-based) => índice 25..34 (0-based).
    3) Fallback: procura padrão textual "Nº 000000000" no topo.
    """
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            if len(doc) == 0:
                return None, None
            txt = doc[0].get_text("text")
            upper = txt.upper()

            # --- 1) Chave de acesso ---
            key = None
            pos = upper.find("CHAVE DE ACESSO")
            if pos != -1:
                tail = upper[pos:pos + 250]  # pega um trecho após a frase
                digits = re.findall(r"\d", tail)
                if len(digits) >= 44:
                    key = "".join(digits[:44])

            # fallback: tentar capturar 44 dígitos próximos em todo o cabeçalho
            if not key:
                # procura blocos de dígitos e espaços que somem 44 dígitos
                # estratégia simples: pega todas as sequências de dígitos; junta as maiores
                digseq = re.findall(r"(\d[ \t]*){44,}", upper)
                if digseq:
                    # remover espaços e ficar com 44
                    only_digits = re.findall(r"\d", digseq[0])
                    if len(only_digits) >= 44:
                        key = "".join(only_digits[:44])

            nf = None
            if key and len(key) == 44:
                # nNF = 9 dígitos, índices 25..34 (0-based)
                nf = key[25:34]

            # --- 2) Fallback textual ---
            if not nf:
                m = re.search(r"\bN[º°o\.]?\s*([0-9]{1,9})\b", upper)
                if m:
                    nf = m.group(1).zfill(9)

            return key, nf
    except Exception:
        return None, None


# ===============================
# Organização para 6 colunas
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
    NF, Desenho, Item Unifilar, NM, Descrição, QTD, Unidade.
    """
    cols = ["NF", "Desenho", "Item Unifilar", "NM", "Descrição", "QTD", "Unidade"]
    df = pd.DataFrame(columns=cols)
    if df_itens.empty:
        return df

    desen = df_itens["COD"].fillna("").astype(str)
    desc  = df_itens["DESCRICAO"].fillna("").astype(str)
    qtd   = df_itens["QTD"].fillna("").astype(str)
    un    = df_itens["UN"].fillna("").astype(str)
    nf    = df_itens["NF"].fillna("").astype(str)

    df["NF"] = nf
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
st.set_page_config(page_title="NF-e (DANFE) — Multi-upload + coluna NF", layout="wide")
st.title("🧾 NF-e (DANFE) — Várias notas + coluna NF por item")

st.markdown(
    """
Carregue **uma ou mais** DANFEs. O app:
- Lê a tabela (pelo **cabeçalho**), consolidando **uma linha por item**;
- Saneia **NCM/SH (8 dígitos)**, **CST (3 dígitos)**, **CFOP (4 dígitos)**;
- Extrai o **nº da NF** do cabeçalho (pela **Chave de Acesso** ou fallback textual) e adiciona a coluna **NF**;
- Mostra:
  1) **Itens consolidados** (com NF, NCM/SH, CST, CFOP, etc.);
  2) **Visão organizada** com **NF, Desenho, Item Unifilar, NM, Descrição, QTD, Unidade**;
- Exporta CSV/XLSX.
"""
)

files = st.file_uploader("Selecione um ou mais PDFs da DANFE", type=["pdf"], accept_multiple_files=True)
btn = st.button("📤 Extrair")

if btn and files:
    all_items = []
    for f in files:
        file_bytes = f.read()

        # Extrai Chave de Acesso e Nº da NF
        chave, nf = extract_access_key_and_nf_number(file_bytes)
        nf_label = nf if nf else "NF_DESCONHECIDA"

        # Extrai itens (consolidados) desta NF
        with st.spinner(f"Lendo itens da NF {nf_label} ({f.name})..."):
            df_items = extract_table_full(file_bytes)

        if df_items.empty:
            st.warning(f"⚠️ Não encontrei itens na NF {nf_label} ({f.name}).")
            continue

        # Anexa colunas de identificação da NF
        df_items.insert(0, "NF", nf_label)
        df_items.insert(1, "Arquivo", f.name)
        df_items.insert(2, "Chave_Acesso", chave or "")

        all_items.append(df_items)

    if not all_items:
        st.error("Nenhuma NF válida encontrada.")
    else:
        df_all = pd.concat(all_items, ignore_index=True)

        st.success(f"Notas processadas: {len(all_items)} — Itens totais: {len(df_all)}")

        st.subheader("1) Itens consolidados (com NF / NCM/SH / CST / CFOP)")
        st.dataframe(df_all, use_container_width=True, height=420)

        # Exports consolidados
        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = df_all.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 CSV (consolidados + NF)", data=csv_bytes, file_name="itens_consolidados_multi_nf.csv", mime="text/csv")
        with c2:
            xls = io.BytesIO()
            with pd.ExcelWriter(xls, engine="openpyxl") as w:
                df_all.to_excel(w, index=False, sheet_name="Consolidados")
            xls.seek(0)
            st.download_button(
                "📥 Excel (consolidados + NF)",
                data=xls,
                file_name="itens_consolidados_multi_nf.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.subheader("2) Visão organizada (NF + 6 colunas solicitadas)")
        df6_parts = []
        for nf_key, df_nf in df_all.groupby("NF", sort=False):
            df6_parts.append(organizar_para_seis_colunas(df_nf))
        df6_all = pd.concat(df6_parts, ignore_index=True)
        st.dataframe(df6_all, use_container_width=True, height=420)

        c3, c4 = st.columns(2)
        with c3:
            csv6 = df6_all.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 CSV (NF + 6 colunas)", data=csv6, file_name="itens_nf_6colunas_multi.csv", mime="text/csv")
        with c4:
            xls6 = io.BytesIO()
            with pd.ExcelWriter(xls6, engine="openpyxl") as w:
                df6_all.to_excel(w, index=False, sheet_name="NF + 6 colunas")
            xls6.seek(0)
            st.download_button(
                "📥 Excel (NF + 6 colunas)",
                data=xls6,
                file_name="itens_nf_6colunas_multi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
