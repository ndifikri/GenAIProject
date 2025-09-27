import streamlit as st

pages = [
    st.Page("supplychain.py", title="Warehouse Optimization Inventory Level"),
    st.Page("marketing.py", title="Aagentic Marketing Assistant")
]

pg = st.navigation(pages=pages, position="sidebar")
pg.run()