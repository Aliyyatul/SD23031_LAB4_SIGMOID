import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function", layout="wide")

st.title("Sigmoid Activation Function")

st.write("Sigmoid is defined as f(x) = 1 / (1 + e⁻ˣ)")

# Sidebar sliders (ADJUSTABLE)
st.sidebar.header("Input Range Settings")
x_min = st.sidebar.slider("Minimum x", -20, -5, -10)
x_max = st.sidebar.slider("Maximum x", 5, 20, 10)

# Generate input
x = np.linspace(x_min, x_max, 400)
y = 1 / (1 + np.exp(-x))

# Plot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output")
ax.set_title("Sigmoid Activation Function")
ax.grid(True)

st.pyplot(fig)

st.write("""
Sigmoid maps input values between 0 and 1.
It is commonly used in binary classification problems.
""")
