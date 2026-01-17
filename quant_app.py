# quant_app.py
import streamlit as st
from auth_gate import require_admin_token

def render(authed_email: str):
    st.markdown("## 中期クオンツ（株価）")
    st.caption(f"認証ユーザー: {authed_email}")
    st.info("LPPL本体をここに移植します（まずは起動確認）。")

def main():
    st.set_page_config(page_title="Quant (Staging)", layout="wide")
    authed_email = require_admin_token()
    render(authed_email)

if __name__ == "__main__":
    main()
