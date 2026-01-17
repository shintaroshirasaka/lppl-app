# fund_app.py
import streamlit as st
from auth_gate import require_admin_token

def render(authed_email: str):
    st.markdown("## 長期ファンダ（財務三表）")
    st.caption(f"認証ユーザー: {authed_email}")
    st.info("まずは年次PL（売上＋利益）から実装します。")

def main():
    st.set_page_config(page_title="Fundamentals (Staging)", layout="wide")
    authed_email = require_admin_token()
    render(authed_email)

if __name__ == "__main__":
    main()
