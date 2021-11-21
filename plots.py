from altair.vegalite.v4.schema.channels import Tooltip
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

data = pd.DataFrame(
   np.random.randn(100,3),
   columns=['a','b','c']
)
# st.line_chart()
fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])
st.pyplot(fig)
chart = alt.Chart(data).mark_circle().encode(
    x='a',y='b'
    )
st.altair_chart(chart)
st.line_chart(data)
st.area_chart(data)
st.bar_chart(data)