"""Upload-only, CPU-hosted tomato detection demonstration."""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import zipfile
from collections import Counter

import streamlit as st

from inference import CHINESE, Detector, annotate, decode_image, jpeg_preview, results_csv

st.set_page_config(page_title="番茄病虫害检测", page_icon=":material/eco:", layout="wide")
st.markdown("""<style>
.stApp { background:#f8faf9; color:#000; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: "SimSun", "Songti SC", "Noto Serif CJK SC", serif; color:#000; }
h1,h2,h3,p,label,[data-testid="stMarkdownContainer"],button {
  font-family: "SimSun", "Songti SC", "Noto Serif CJK SC", serif !important;
  color:#000; letter-spacing:0; }
.block-container { max-width:1440px; padding-top:2rem; }
h1 { font-size:2rem !important; }
h2 { font-size:1.3rem !important; }
[data-testid="stMetricValue"] { font-size:1.7rem; }
[data-testid="stSidebar"] { background:#eef3f1; }
button { border-radius:6px !important; }
h3 { font-size:1.1rem !important; overflow-wrap:anywhere; }
button[kind="primaryFormSubmit"] { background:#dcefe6; color:#000; border-color:#18755f; }
</style>""", unsafe_allow_html=True)


def authenticate():
    password = os.getenv("DEMO_ACCESS_CODE", "")
    if not password or st.session_state.get("authenticated"):
        return
    st.title("番茄病虫害检测")
    with st.form("access"):
        entered = st.text_input("访问口令", type="password")
        if st.form_submit_button("进入", icon=":material/login:"):
            if hmac.compare_digest(entered.encode(), password.encode()):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("访问口令不正确。")
    st.stop()


authenticate()


@st.cache_resource
def detector():
    return Detector()


models = {"GT-RFSKD · 轻量模型": "gtrfskd"}
if os.getenv("ENABLE_P2D", "0") == "1":
    models["P2-DySample · 高精度模型"] = "p2d"

with st.sidebar:
    st.header("检测设置")
    with st.form("detect"):
        model_label = st.selectbox("检测模型", list(models))
        confidence = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("NMS 阈值", 0.1, 0.95, 0.7, 0.05)
        files = st.file_uploader("待检测图像", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("开始检测", type="primary", icon=":material/search:", use_container_width=True)
    st.caption("640 × 640 · ONNX Runtime · CPU")
    st.caption("最多 2 张，每张不超过 2 MiB、400 万像素。")
    if st.button("清空结果", icon=":material/delete:", use_container_width=True):
        st.session_state.pop("artifacts", None)
        st.session_state.pop("errors", None)
    source = os.getenv("SOURCE_URL", "")
    if source.startswith("https://github.com/"):
        st.link_button("源代码与许可", source, icon=":material/code:")

st.title("番茄病虫害检测")
st.caption("六类叶片目标 · 图像检测与结果导出")

if submitted:
    st.session_state["artifacts"] = []
    st.session_state["errors"] = []
    if not files:
        st.warning("请先选择图片。")
    elif len(files) > 2:
        st.error("每批最多 2 张图片。")
    else:
        with st.spinner("检测中"):
            for index, uploaded in enumerate(files):
                try:
                    rgb = decode_image(uploaded.getvalue())
                    found, timing = detector().predict(rgb, models[model_label], confidence, iou)
                    preview = annotate(rgb, found)
                    st.session_state["artifacts"].append({
                        "id": f"{index}-{hashlib.sha256(uploaded.getvalue()).hexdigest()[:10]}",
                        "name": uploaded.name, "model": models[model_label], "label": model_label,
                        "confidence": confidence, "iou": iou, "detections": found, "timing": timing,
                        "original": jpeg_preview(rgb), "preview": jpeg_preview(preview),
                    })
                    del rgb, preview
                except (ValueError, RuntimeError) as exc:
                    st.session_state["errors"].append(f"第 {index+1} 张：{exc}")
                except Exception:
                    st.session_state["errors"].append(f"第 {index+1} 张检测失败，请联系维护人员检查模型配置。")

for error in st.session_state.get("errors", []):
    st.error(error)
artifacts = st.session_state.get("artifacts", [])
if not artifacts:
    st.subheader("检测工作区")
    st.info("尚无检测结果")
    st.divider()
    st.subheader("检测类别")
    columns = st.columns(3)
    for index, name in enumerate(CHINESE):
        with columns[index % 3]:
            st.write(name)
else:
    detections = [d for item in artifacts for d in item["detections"]]
    cols = st.columns(4)
    cols[0].metric("已处理图像", len(artifacts))
    cols[1].metric("检出目标", len(detections))
    cols[2].metric("涉及类别", len({d["class_id"] for d in detections}))
    cols[3].metric("平均推理耗时", f'{sum(a["timing"]["inference_ms"] for a in artifacts)/len(artifacts):.1f} ms')
    st.divider()
    image_tab, table_tab = st.tabs(["检测图像", "检测明细"])
    with image_tab:
        for artifact in artifacts:
            st.subheader(artifact["name"])
            st.caption(f'{artifact["label"]} · 置信度 {artifact["confidence"]:.2f} · NMS {artifact["iou"]:.2f}')
            left, right = st.columns(2)
            left.image(artifact["original"], caption="原始图像", use_container_width=True)
            right.image(artifact["preview"], caption=f'检测结果 · {len(artifact["detections"])} 个目标', use_container_width=True)
            counts = Counter(d["class_cn"] for d in artifact["detections"])
            st.write("　".join(f"{name} {count}" for name, count in counts.items()) or "未检出高于当前置信度阈值的目标")
            timing = artifact["timing"]
            st.caption(f'预处理 {timing["preprocess_ms"]:.1f} ms · 推理 {timing["inference_ms"]:.1f} ms · 后处理 {timing["postprocess_ms"]:.1f} ms')
            st.download_button("下载检测图", artifact["preview"], f'detection-{artifact["id"]}.jpg',
                               "image/jpeg", icon=":material/download:", key=f'download-{artifact["id"]}')
    with table_tab:
        rows = []
        for artifact in artifacts:
            for i, det in enumerate(artifact["detections"], 1):
                rows.append({"图像": artifact["name"], "目标": i, "类别": det["class_cn"],
                             "置信度": round(det["confidence"], 4),
                             **dict(zip(["左", "上", "右", "下"], [round(x, 1) for x in det["box"]]))})
        st.dataframe(rows, hide_index=True, use_container_width=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("detections.csv", results_csv(artifacts))
        for artifact in artifacts:
            archive.writestr(f'detection-{artifact["id"]}.jpg', artifact["preview"])
    st.download_button("导出本批结果", buffer.getvalue(), "tomato-detections.zip", "application/zip", icon=":material/download:")

st.divider()
st.caption("检测结果仅供研究与辅助观察，不替代植保诊断。图片不写入持久存储；刷新或重新提交后结果可能清除。")
