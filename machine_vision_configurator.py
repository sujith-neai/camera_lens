import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Machine Vision Configurator", layout="wide")

# === FORCE DEFAULT TEXT WRAPPING ===
st.markdown("""
<style>
    .stDataFrame {
        width: 100% !important;
        height: auto !important;
        margin-bottom: 0 !important;
    }
    .stDataFrame > div > div > div > div > div {
        height: auto !important;
    }
    [role="rowgroup"] > [role="row"] {
        height: auto !important;
    }
    [role="grid"] {
        height: auto !important;
        margin-bottom: 0 !important;
    }
    .stDataFrame td {
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.4 !important;
        padding: 15px 12px !important;
        max-width: none !important;
        overflow: visible !important;
        text-overflow: clip !important;
        vertical-align: top !important;
        height: auto !important;
    }
    .stDataFrame th {
        white-space: normal !important;
        word-break: break-word !important;
        padding: 12px 10px !important;
        overflow: visible !important;
    }
    [data-testid="arrowTable"] {
        max-height: none !important;
        margin-bottom: 0 !important;
    }
    [data-testid="stDataFrame"] {
        margin-bottom: 0 !important;
    }
    .stMetric {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Machine Vision System Configurator")

# ====================== LOAD CONFIG ======================
CONFIG_PATH = Path("machine_vision_config.json")

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        st.error("❌ machine_vision_config.json not found!")
        st.stop()

config = load_config()

if st.button("🔄 Reload Config"):
    config = load_config()
    st.success("✅ Config reloaded!")

# ====================== SIDEBAR ======================
st.sidebar.header("Use Case Quick Load")
use_case = st.sidebar.selectbox("Load example", list(config["use_cases"].keys()))

# ====================== MAIN INPUTS ======================
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📐 Object Geometry")
    uc = config["use_cases"][use_case]
    obj_width = st.number_input("Object Width (mm)", value=float(uc["width_mm"]), min_value=1.0, step=1.0)
    obj_height = st.number_input("Object Height (mm)", value=float(uc["height_mm"]), min_value=1.0, step=1.0)
    wd = st.number_input("Working Distance (mm)", value=float(uc["wd_mm"]), min_value=50.0, step=10.0)
    depth_var = st.slider("Depth variation (mm)", 0.0, 20.0, float(uc["depth_variation_mm"]), step=0.1)

with col2:
    st.subheader("🔍 Inspection Requirements")
    task_options = list(config["task_types"].keys())
    task_type = st.selectbox("Inspection Type", task_options, index=task_options.index(uc["task_preset"]))
    default_px = config["task_types"][task_type]
    
    # OCR checkbox to increase pixel requirements
    enable_ocr = st.checkbox("Enable OCR (increases pixel requirement by 40%)", value=False)
    ocr_multiplier = 1.4 if enable_ocr else 1.0
    adjusted_default_px = int(default_px * ocr_multiplier)
    
    px_across_feature = st.slider("Required pixels across smallest feature", 
                                  config["defaults"]["px_slider_min"], config["defaults"]["px_slider_max"], adjusted_default_px, step=1)
    smallest_feature_mm = st.number_input("Smallest feature size (mm)", value=float(uc["smallest_feature_mm"]), min_value=0.1, step=0.05)
    mono_vs_color = st.radio("Sensor type", ["Monochrome (recommended)", "Color"], horizontal=True)

# ====================== MANUAL OVERRIDES ======================
st.subheader("🔧 Manual Overrides (leave at 0 for auto-calculation)")
col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
hfov_manual = col_ov1.number_input("Manual Horizontal FOV (°)", value=0.0, min_value=0.0, step=0.1)
vfov_manual = col_ov2.number_input("Manual Vertical FOV (°)", value=0.0, min_value=0.0, step=0.1)
manual_px_per_mm = col_ov3.number_input("Manual Pixels per mm", value=0.0, min_value=0.0, step=0.1)
manual_dof = col_ov4.number_input("Manual DOF (mm)", value=0.0, min_value=0.0, step=0.1)
f_number = st.selectbox("f-number for DOF", [4.0, 5.6, 8.0, 11.0, 16.0], index=2)

# ====================== CALCULATIONS ======================
st.divider()
st.subheader("📊 Live Calculations & Results")

px_per_mm_calc = px_across_feature / smallest_feature_mm
hfov_calc = 2 * math.degrees(math.atan(obj_width / (2 * wd))) if wd > 0 else 0
vfov_calc = 2 * math.degrees(math.atan(obj_height / (2 * wd))) if wd > 0 else 0
required_px_mm = manual_px_per_mm if manual_px_per_mm > 0 else px_per_mm_calc
hfov = hfov_manual if hfov_manual > 0 else hfov_calc
vfov = vfov_manual if vfov_manual > 0 else vfov_calc

horiz_px = int(np.ceil(obj_width * required_px_mm))
vert_px = int(np.ceil(obj_height * required_px_mm))
min_mp = round(horiz_px * vert_px / 1_000_000, 2)
magnification = required_px_mm * 0.001

focal_lengths = {}
for name, sw in config["sensor_formats"].items():
    f = (sw * wd) / obj_width if wd > 0 else 0
    focal_lengths[name] = round(f, 2)

# Auto-select best sensor format (closest to practical range 12-50mm)
practical_range = (12, 50)
best_sensor = min(focal_lengths.keys(), 
                  key=lambda x: abs(focal_lengths[x] - sum(practical_range)/2) 
                                if practical_range[0] <= focal_lengths[x] <= practical_range[1]
                                else float('inf'))
# Fallback if none in range
if abs(focal_lengths[best_sensor] - sum(practical_range)/2) == float('inf'):
    best_sensor = min(focal_lengths.keys(), key=lambda x: abs(focal_lengths[x] - 25))

selected_sensor = best_sensor
suggested_focal = focal_lengths[selected_sensor]

coc = 0.01
dof_calc = round((2 * f_number * coc * (wd ** 2)) / (suggested_focal ** 2), 1) if suggested_focal > 0 else 0
dof = manual_dof if manual_dof > 0 else dof_calc

standard_camera_mp = [2, 5, 8, 12, 20, 25, 45]
rec_camera_mp = next((t for t in standard_camera_mp if t >= min_mp), standard_camera_mp[-1])

# Find nearest standard available focal lengths for C-mount lenses
standard_focal_lengths = [8, 12, 16, 25, 35, 50, 75, 100]
sorted_by_distance = sorted(standard_focal_lengths, key=lambda x: abs(x - suggested_focal))
nearest_focal = sorted_by_distance[0]
second_nearest_focal = sorted_by_distance[1] if len(sorted_by_distance) > 1 else None

# ====================== KEY METRICS ======================
colA, colB, colC, colD, colE, colF = st.columns(6)
colA.metric("Horizontal FOV", f"{hfov:.1f}°", f"Calc: {hfov_calc:.1f}°" if hfov_manual > 0 else "")
colB.metric("Vertical FOV", f"{vfov:.1f}°", f"Calc: {vfov_calc:.1f}°" if vfov_manual > 0 else "")
colC.metric("Required Resolution", f"{horiz_px}×{vert_px}px", f"{min_mp} MP")
colD.metric("Pixels per mm", f"{required_px_mm:.2f}")
colE.metric("Magnification", f"{magnification:.5f}")
colF.metric("Depth of Field", f"{dof:.1f} mm", f"Calc: {dof_calc:.1f} mm" if manual_dof > 0 else "")

# ====================== RECOMMENDATIONS WITH FULL DEFAULT WRAPPING ======================
st.subheader("🏭 Recommended Hardware (India-available - Best Performance)")

# Display focal length recommendation with available options
lens_recommendation = f"📐 **Calculated Focal Length: {suggested_focal} mm** (Sensor: {selected_sensor})\n\n"
lens_recommendation += f"**🛒 Recommended Available Lenses:**\n"
lens_recommendation += f"- **Primary: {nearest_focal} mm** (difference: {abs(nearest_focal - suggested_focal):.2f} mm)\n"
if second_nearest_focal:
    lens_recommendation += f"- **Alternative: {second_nearest_focal} mm** (difference: {abs(second_nearest_focal - suggested_focal):.2f} mm)"

st.info(lens_recommendation)

rec_data = []
for item in config["hardware_recommendations"]:
    # Skip Software/Library component
    if item["component"].lower() == "software / library":
        continue
    template = item["template"].replace("{focal}", str(suggested_focal)).replace("{mp}", str(min_mp))
    expl = item["explanation"] \
        .replace("{hfov}", f"{hfov:.1f}°") \
        .replace("{mp}", f"{min_mp}") \
        .replace("{rec_mp}", str(rec_camera_mp)) \
        .replace("{task}", task_type)
    rec_data.append([item["component"], template, expl])
    # Limit to 3 recommendations only
    if len(rec_data) >= 3:
        break

rec_df = pd.DataFrame(rec_data, columns=["Component", "Recommendation", "Why This?"])

st.dataframe(
    rec_df,
    use_container_width=True,
    height=320,
    hide_index=True,
    column_config={
        "Component": st.column_config.TextColumn("Component"),
        "Recommendation": st.column_config.TextColumn("Recommendation"),
        "Why This?": st.column_config.TextColumn("Why This?")
    }
)

# ====================== COMPREHENSIVE SUMMARY ======================
st.divider()
st.subheader("📋 Configuration Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.write("**📥 INPUT PARAMETERS**")
    st.write(f"""
    • **Use Case:** {use_case}
    • **Object Size:** {obj_width} × {obj_height} mm
    • **Working Distance:** {wd} mm
    • **Smallest Feature:** {smallest_feature_mm} mm
    • **Task Type:** {task_type}
    • **Depth Variation:** {depth_var} mm
    • **OCR Enabled:** {'Yes' if enable_ocr else 'No'}
    • **Sensor Type:** {mono_vs_color}
    """)

with summary_col2:
    st.write("**📊 CALCULATED OUTPUTS**")
    st.write(f"""
    • **Horizontal FOV:** {hfov:.1f}°
    • **Vertical FOV:** {vfov:.1f}°
    • **Pixels/mm Required:** {required_px_mm:.2f}
    • **Required Resolution:** {horiz_px} × {vert_px} px
    • **Required MP:** {min_mp} MP
    • **Magnification:** {magnification:.5f}
    • **Depth of Field:** {dof:.1f} mm
    • **f-number:** {f_number}
    """)

with summary_col3:
    st.write("**🎯 RECOMMENDED SYSTEM**")
    st.write(f"""
    • **Sensor Format:** {selected_sensor}
    • **Ideal Focal Length:** {suggested_focal} mm
    • **Primary Lens:** {nearest_focal} mm
    • **Alt. Lens:** {second_nearest_focal} mm
    • **Camera:** 12–25 MP GigE
    • **Lighting:** Diffuse/Coaxial LED
    • **Interface:** GigE Vision
    • **Region:** India-available
    """)

# ====================== VISUALIZATIONS ======================
st.subheader("📈 Visualizations")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot([-obj_width/2, 0, obj_width/2], [0, -wd, 0], 'b-', lw=3)
ax[0].text(0, -wd/2, f"{hfov:.0f}°", ha='center', fontsize=14, color='blue')
ax[0].set_title("Field of View Cone")
ax[0].set_xlabel("Object plane (mm)")
ax[0].set_ylabel("Working Distance (mm)")
ax[0].invert_yaxis()
ax[0].grid(True)

x = np.linspace(0, obj_width, 20)
y = np.linspace(0, obj_height, 20)
X, Y = np.meshgrid(x, y)
im = ax[1].imshow(np.full_like(X, required_px_mm), extent=[0, obj_width, 0, obj_height], origin='lower', cmap='viridis')
ax[1].set_title("Resolution density across object (px/mm)")
plt.colorbar(im, ax=ax[1], label="px/mm")
ax[1].set_xlabel("Width (mm)")
ax[1].set_ylabel("Height (mm)")
st.pyplot(fig)

# ====================== MOVING CONVEYOR MODULE ======================
st.divider()
st.subheader("🏭 Moving Conveyor Module")

conveyor_enabled = st.checkbox("Enable Moving Conveyor Analysis", value=False)

if conveyor_enabled:
    conv_col1, conv_col2, conv_col3 = st.columns(3)

    with conv_col1:
        speed_unit = st.selectbox("Speed unit", ["mm/s", "m/min"])
        speed_default = 200.0 if speed_unit == "mm/s" else 12.0
        conveyor_speed_input = st.number_input(f"Conveyor speed ({speed_unit})", value=speed_default, min_value=1.0, step=1.0)
        conveyor_speed_mm_s = conveyor_speed_input if speed_unit == "mm/s" else conveyor_speed_input * 1000 / 60

    with conv_col2:
        acceptable_blur_px = st.number_input("Acceptable motion blur (px)", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
        object_gap_mm = st.number_input("Gap between objects on conveyor (mm)", value=50.0, min_value=0.0, step=5.0)

    with conv_col3:
        frame_overlap_pct = st.slider("Frame overlap (%)", 0, 50, 20, step=5)
        trigger_method = st.selectbox("Trigger method", ["Encoder", "PLC Timer", "Photoelectric Sensor"])

    # Core calculations
    # Strobe duration is the effective shutter for motion freeze
    max_strobe_us = (acceptable_blur_px / (conveyor_speed_mm_s * required_px_mm)) * 1e6 if conveyor_speed_mm_s > 0 and required_px_mm > 0 else 0
    # Camera exposure set ~3× strobe to guarantee strobe fires within window
    recommended_exposure_ms = min(max_strobe_us * 3 / 1000, 5.0)

    frame_step_mm = obj_height * (1 - frame_overlap_pct / 100)
    min_fps = conveyor_speed_mm_s / frame_step_mm if frame_step_mm > 0 else 0
    trigger_interval_ms = (frame_step_mm / conveyor_speed_mm_s) * 1000 if conveyor_speed_mm_s > 0 else 0

    cycle_mm = obj_height + object_gap_mm
    throughput_per_min = (conveyor_speed_mm_s * 60) / cycle_mm if cycle_mm > 0 else 0

    st.markdown("---")
    conv_m1, conv_m2, conv_m3, conv_m4, conv_m5 = st.columns(5)
    conv_m1.metric("Conveyor Speed", f"{conveyor_speed_mm_s:.0f} mm/s", f"{conveyor_speed_mm_s * 60 / 1000:.1f} m/min")
    conv_m2.metric("Max Strobe Duration", f"{max_strobe_us:.1f} µs")
    conv_m3.metric("Camera Exposure", f"{recommended_exposure_ms:.3f} ms")
    conv_m4.metric("Min Frame Rate", f"{min_fps:.1f} fps")
    conv_m5.metric("Throughput", f"{throughput_per_min:.0f} obj/min")

    conv_m6, conv_m7 = st.columns(2)
    conv_m6.metric("Trigger Interval", f"{trigger_interval_ms:.1f} ms")
    conv_m7.metric("Frame Step on Belt", f"{frame_step_mm:.1f} mm")

    if max_strobe_us < 50:
        st.warning(f"⚠️ Very short strobe ({max_strobe_us:.1f} µs) required — high-power strobe essential. Consider reducing speed or relaxing blur tolerance.")
    elif max_strobe_us < 200:
        st.info(f"ℹ️ Short strobe ({max_strobe_us:.1f} µs) — use a high-power Q-series LED strobe controller.")
    else:
        st.success(f"✅ Comfortable strobe duration ({max_strobe_us:.0f} µs) — standard strobed lighting should work well.")

    if min_fps > 100:
        st.warning(f"⚠️ High frame rate required ({min_fps:.0f} fps) — verify camera supports this at full resolution or reduce overlap.")

    # Conveyor synchronization hardware recommendations
    st.markdown("**🔌 Conveyor Synchronisation Hardware**")
    conv_hw_data = []

    if trigger_method == "Encoder":
        conv_hw_data.append([
            "Encoder",
            "Incremental rotary encoder (500–2000 PPR) on drive roller",
            f"Trigger camera every {frame_step_mm:.1f} mm of belt travel (every {trigger_interval_ms:.1f} ms at current speed). Configure encoder controller to count pulses/mm and output a GPIO pulse at each {frame_step_mm:.1f} mm interval."
        ])
    elif trigger_method == "PLC Timer":
        conv_hw_data.append([
            "PLC / Timer",
            f"PLC digital output at {min_fps:.1f} Hz ({trigger_interval_ms:.1f} ms period)",
            f"Set PLC high-speed output to pulse every {trigger_interval_ms:.1f} ms to step the camera at {frame_step_mm:.1f} mm intervals. Recalibrate PLC timer if conveyor speed changes."
        ])
    else:
        conv_hw_data.append([
            "Photoelectric Sensor",
            "Diffuse reflection sensor at conveyor entry point",
            f"Detects leading edge of each object and triggers a capture burst at {min_fps:.1f} fps until object clears FOV. Follow initial trigger with a {trigger_interval_ms:.1f} ms periodic timer for subsequent frames."
        ])

    conv_hw_data.append([
        "Strobe Controller",
        f"High-power LED strobe — pulse width ≤ {max_strobe_us:.0f} µs",
        f"Fires within the camera exposure window ({recommended_exposure_ms:.3f} ms) to freeze motion at {conveyor_speed_mm_s:.0f} mm/s. Q-Series or Gardasoft RT-series recommended for sub-100 µs pulses."
    ])

    conv_hw_data.append([
        "Camera Trigger Input",
        "Hardware GPIO / GigE trigger (not software trigger)",
        f"Camera must respond to hardware trigger within < {max_strobe_us / 2:.0f} µs jitter. Set exposure to {recommended_exposure_ms:.3f} ms. Verify camera's max trigger rate ≥ {min_fps:.1f} fps at current resolution ({horiz_px}×{vert_px} px)."
    ])

    conv_df = pd.DataFrame(conv_hw_data, columns=["Component", "Recommendation", "Why This?"])
    st.dataframe(
        conv_df,
        use_container_width=True,
        height=220,
        hide_index=True,
        column_config={
            "Component": st.column_config.TextColumn("Component"),
            "Recommendation": st.column_config.TextColumn("Recommendation"),
            "Why This?": st.column_config.TextColumn("Why This?")
        }
    )

    st.info(f"""
**📦 Conveyor Summary** — at **{conveyor_speed_mm_s:.0f} mm/s** with **{required_px_mm:.2f} px/mm** resolution:
- Freeze motion to **{acceptable_blur_px} px** blur → max strobe **{max_strobe_us:.1f} µs**, camera exposure **{recommended_exposure_ms:.3f} ms**
- Capture **{min_fps:.1f} fps** with **{frame_overlap_pct}% overlap** (frame step = **{frame_step_mm:.1f} mm**)
- Expected throughput: **{throughput_per_min:.0f} parts/minute**
    """)

# ====================== SAVE CONFIGURATION ======================
st.divider()
st.subheader("💾 Save Configuration to Library")

save_col1, save_col2 = st.columns([3, 1])
with save_col1:
    config_name = st.text_input("Give your configuration a name:", placeholder="e.g., PCB_High_Speed_Inspection", key="config_name_input")
with save_col2:
    save_button = st.button("Save to Config", use_container_width=True)

if save_button and config_name:
    # Create the new use case entry
    new_use_case = {
        "width_mm": obj_width,
        "height_mm": obj_height,
        "wd_mm": wd,
        "smallest_feature_mm": smallest_feature_mm,
        "depth_variation_mm": depth_var,
        "task_preset": task_type
    }
    
    # Load current config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        current_config = json.load(f)
    
    # Add the new use case
    current_config["use_cases"][config_name] = new_use_case
    
    # Save back to file
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current_config, f, indent=2)
    
    st.success(f"✅ Configuration '{config_name}' saved! It will appear in the 'Load example' dropdown on next refresh.")
    st.info("🔄 Reload the page to see your new configuration in the sidebar dropdown.")

elif save_button and not config_name:
    st.warning("⚠️ Please enter a configuration name first!")

st.caption("✅ All text now wraps automatically by default • No double-click required • Resolution shows MP clearly • Built for Sujith, Bengaluru")
