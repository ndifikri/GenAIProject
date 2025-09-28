import streamlit as st

pages = [
    st.Page("supplychain.py", title="Warehouse Optimization Inventory Level"),
    st.Page("marketing.py", title="Agentic Marketing Assistant"),
    st.Page("product_design.py", title="Agentic Product Design Assistant")
]

pg = st.navigation(pages=pages, position="sidebar")
pg.run()