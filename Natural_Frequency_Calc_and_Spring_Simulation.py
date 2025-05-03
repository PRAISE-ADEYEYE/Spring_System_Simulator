import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import pandas as pd

# 🎨 Neon Color Palette
NEON_BLUE = "#00FFFF"
NEON_PINK = "#FF6EC7"
NEON_YELLOW = "#FFFF33"
NEON_PURPLE = "#DA70D6"
NEON_GREEN = "#39FF14"
BLACK_BG = "#0d0d0d"
WHITE = "#ffffff"


# 💡 Function to calculate natural frequency
def get_frequency(k, m):
    return np.sqrt(k / m) / (2 * np.pi)


# 🎯 Function to animate oscillation
def animate_oscillation(k, m, ax, spring_length=1.0, damping=0.0):
    time = np.linspace(0, 10, 100)
    displacement = np.cos(2 * np.pi * get_frequency(k, m) * time) * np.exp(-damping * time)

    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Oscillation of the Spring-Mass System", color=NEON_YELLOW)

    line, = ax.plot([], [], lw=2, color=NEON_BLUE)

    def update(i):
        line.set_data([0, displacement[i]], [0, 0])
        return line,

    ani = FuncAnimation(ax.figure, update, frames=len(time), interval=50, blit=True)
    return ani


# 🎯 Function to simulate damped and forced oscillations
def damped_forced_oscillation(m, k, c, F, omega_d, t):
    def model(y, t, m, k, c, F, omega_d):
        x, v = y
        dxdt = v
        dvdt = (-c * v - k * x + F * np.cos(omega_d * t)) / m
        return [dxdt, dvdt]

    y0 = [0.0, 0.0]  # Initial conditions: x=0, v=0
    sol = odeint(model, y0, t, args=(m, k, c, F, omega_d))
    return sol[:, 0]  # Return displacement


# 🌟 Streamlit UI Integration
st.set_page_config(page_title="Advanced Spring System Simulator", page_icon="⚙️", layout="wide")

# 📢 Streamlit Sidebar for user input
st.sidebar.header("💥 Spring System Parameters 💥")
m1 = st.sidebar.number_input("Mass 1 (kg) for Single Spring System", min_value=0.1, max_value=10.0, value=2.0)
k1 = st.sidebar.number_input("Spring Constant (N/m) for Single Spring System", min_value=100, max_value=5000, value=800)
m2 = st.sidebar.number_input("Mass 2 (kg) for Two Springs in Parallel", min_value=0.1, max_value=10.0, value=1.4)
k2_1 = st.sidebar.number_input("Spring Constant K1 (N/m) for Parallel Springs", min_value=100, max_value=5000,
                               value=4000)
k2_2 = st.sidebar.number_input("Spring Constant K2 (N/m) for Parallel Springs", min_value=100, max_value=5000,
                               value=1600)
c1 = st.sidebar.slider("Damping Coefficient (kg/s)", 0.1, 10.0, 0.5)
F1 = st.sidebar.slider("Force Amplitude (N)", 0.1, 50.0, 5.0)
omega_d = st.sidebar.slider("Driving Frequency (Hz)", 0.5, 10.0, 1.0)

# 📈 Calculating Natural Frequencies for both systems
f1 = get_frequency(k1, m1)
f2 = get_frequency(k2_1 + k2_2, m2)

# 📢 Display Frequencies in Streamlit
st.header("🎶 Natural Frequency Results")
st.write(f"✅ Single Spring System Frequency: {f1:.2f} Hz")
st.write(f"✅ Two Parallel Springs Frequency: {f2:.2f} Hz")

# 📊 Interactive Graphs with Streamlit
st.header("📊 Frequency vs Parameters")
st.write("Below are the graphs showing how frequency varies with mass and stiffness:")

# 🧲 Frequency vs Mass
mass_range = np.linspace(0.5, 10, 100)
freq_vs_mass = get_frequency(k1, mass_range)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(mass_range, freq_vs_mass, color=NEON_PINK, linewidth=3)
ax1.set_title("🌀 Frequency vs Mass", color=NEON_YELLOW, fontsize=14)
ax1.set_xlabel("Mass (kg)", color=WHITE)
ax1.set_ylabel("Frequency (Hz)", color=WHITE)
ax1.set_facecolor(BLACK_BG)
ax1.tick_params(colors=WHITE)
ax1.grid(True, linestyle=':', alpha=0.4)
st.pyplot(fig1)

# 🧲 Frequency vs Stiffness
stiffness_range = np.linspace(100, 5000, 100)
freq_vs_stiff = get_frequency(stiffness_range, m1)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(stiffness_range, freq_vs_stiff, color=NEON_GREEN, linewidth=3)
ax2.set_title("🧲 Frequency vs Stiffness", color=NEON_YELLOW, fontsize=14)
ax2.set_xlabel("Stiffness (N/m)", color=WHITE)
ax2.set_ylabel("Frequency (Hz)", color=WHITE)
ax2.set_facecolor(BLACK_BG)
ax2.tick_params(colors=WHITE)
ax2.grid(True, linestyle=':', alpha=0.4)
st.pyplot(fig2)

# 🧷 Animation of Oscillation with Damping and Force
st.header("🔄 Damped and Forced Oscillation Animation")
st.write("Watch the oscillation of the spring-mass system under damping and driving force.")
time = np.linspace(0, 10, 1000)
displacement = damped_forced_oscillation(m1, k1, c1, F1, omega_d, time)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(time, displacement, color=NEON_BLUE)
ax3.set_title("Damped and Forced Oscillation", color=NEON_YELLOW)
ax3.set_xlabel("Time (s)", color=WHITE)
ax3.set_ylabel("Displacement (m)", color=WHITE)
ax3.set_facecolor(BLACK_BG)
ax3.tick_params(colors=WHITE)
ax3.grid(True, linestyle=':', alpha=0.4)
st.pyplot(fig3)

# 💾 Option to save time-domain graph
st.write("💾 Save the Time-Domain Graph")
save_time_graph = st.radio("Would you like to save the time-domain graph?", ["Yes", "No"])

if save_time_graph == "Yes":
    fig3.savefig("time_domain_graph.png")
    st.write("Graph saved successfully!")

# 3D Visualization of Spring System
st.header("🔵 3D Visualization of Spring-Mass System")
st.write("Visualize the spring system in 3D space to better understand its motion.")

# 3D plotting can be done with matplotlib (though limited for complex simulations)
fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111, projection='3d')
x = np.linspace(-1, 1, 100)
y = np.cos(2 * np.pi * x)  # Simulate oscillation
z = np.sin(2 * np.pi * x)
ax4.plot(x, y, z, color=NEON_PINK)
ax4.set_title("3D Spring System Motion", color=NEON_YELLOW)
ax4.set_xlabel("X", color=WHITE)
ax4.set_ylabel("Y", color=WHITE)
ax4.set_zlabel("Z", color=WHITE)
st.pyplot(fig4)

# 📈 Sensitivity Analysis
st.header("📉 Parameter Sensitivity Analysis")
st.write("Observe how small changes in mass or spring constant affect the system's frequency.")

sensitivity_range = np.linspace(0.5, 10, 100)
sensitivity_freq = get_frequency(k1, sensitivity_range)

fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.plot(sensitivity_range, sensitivity_freq, color=NEON_GREEN, linewidth=3)
ax5.set_title("Sensitivity of Frequency to Mass", color=NEON_YELLOW, fontsize=14)
ax5.set_xlabel("Mass (kg)", color=WHITE)
ax5.set_ylabel("Frequency (Hz)", color=WHITE)
ax5.set_facecolor(BLACK_BG)
ax5.tick_params(colors=WHITE)
ax5.grid(True, linestyle=':', alpha=0.4)
st.pyplot(fig5)

# 📈 Export Data to CSV
st.header("💾 Export Simulation Data")
st.write("You can export the simulation data to a CSV file for further analysis.")

data = {
    "Mass (kg) for Single Spring": [m1],
    "Spring Constant (N/m) for Single Spring": [k1],
    "Frequency (Hz) for Single Spring": [f1],
    "Mass (kg) for Parallel Springs": [m2],
    "Spring Constant K1 (N/m) for Parallel Springs": [k2_1],
    "Spring Constant K2 (N/m) for Parallel Springs": [k2_2],
    "Frequency (Hz) for Parallel Springs": [f2],
    "Damping Coefficient (kg/s)": [c1],
    "Force Amplitude (N)": [F1],
    "Driving Frequency (Hz)": [omega_d]
}

df = pd.DataFrame(data)
csv = df.to_csv(index=False)

st.download_button(
    label="Download Simulation Data",
    data=csv,
    file_name="simulation_data.csv",
    mime="text/csv"
)
# ================================
# 💥💫 DEVELOPED BY PRAISE ADEYEYE 💫💥
# ================================

st.markdown("""
    <div style="text-align: center; font-size: 20px; color: #39FF14;">
        🔥💡🎉 <b>Developed by</b> <span style="color: #FF6EC7;">Praise Adeyeye</span> 🎉💡🔥
        <br><br>
        🚀💥💻🌟 <i>Where innovation meets engineering excellence!</i> 🌟💻💥🚀
        <br><br>
        ⚡️💎🛸 Stay inspired, stay bold! ⚡️💎🛸
    </div>
""", unsafe_allow_html=True)
