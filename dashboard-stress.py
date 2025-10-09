import streamlit as lit
import pandas as pd

lit.title("👋 Olá a todos, nosso primeiro dashboard:")

df = pd.read_csv("data/academic-stress-level.csv")


lit.write("DataFrame using st.write() function")
#lit.write(df.head())
lit.dataframe(df.head())
#lit.table(df.head())

#lit.scatter_chart(df)