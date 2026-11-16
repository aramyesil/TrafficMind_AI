"""TrafficMind AI traffic analysis interface."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import plotly.express as px
import streamlit as st

from src.labels import CLASS_COLORS, class_label
from src.training_summary import best_validation_metrics
from src.video_analysis import (
    MODEL_PATH,
    PROJECT_ROOT,
    analyze_video,
    draw_counting_line,
    ensure_decodable_video,
    line_pixel,
    load_model,
    read_first_frame,
)

RESULTS_CSV = PROJECT_ROOT / "runs" / "detect" / "runs" / "yolo26n" / "results.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
UPLOAD_ROOT = OUTPUT_ROOT / "uploads"

st.set_page_config(page_title="TrafficMind AI", page_icon="🚦", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e7eb;
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_model():
    return load_model(MODEL_PATH)


@st.cache_data
def cached_metrics(csv_path: str) -> dict:
    return best_validation_metrics(Path(csv_path))


def save_upload(uploaded) -> Path:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_ROOT / Path(uploaded.name).name
    destination.write_bytes(uploaded.getbuffer())
    return destination


def preview_image(frame, orientation: str, line_ratio: float):
    height, width = frame.shape[:2]
    position = line_pixel(orientation, line_ratio, width, height)
    preview = draw_counting_line(frame, orientation, position)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def chart_layout(figure):
    figure.update_layout(
        margin=dict(l=8, r=8, t=48, b=8),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif", color="#1f2933"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#e4e7eb")
    return figure


def render_results(result) -> None:
    peak, total, frames = st.columns(3)
    peak.metric("En yüksek anlık araç", result.instant_peak)
    total.metric("Toplam geçen araç", result.total_crossings)
    frames.metric("İşlenen kare", result.frame_count)

    st.subheader("İşlenmiş video")
    st.video(str(result.output_video))
    video_bytes = result.output_video.read_bytes()
    download_col, csv_col = st.columns(2)
    download_col.download_button(
        "Videoyu indir",
        data=video_bytes,
        file_name=result.output_video.name,
        mime="video/mp4",
    )
    stats_csv = result.frame_stats.to_csv(index=False).encode("utf-8-sig")
    csv_col.download_button(
        "Kare istatistiklerini indir",
        data=stats_csv,
        file_name="frame_stats.csv",
        mime="text/csv",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Anlık yoğunluk")
        timeline = px.line(
            result.frame_stats,
            x="Zaman (sn)",
            y="Anlık araç",
            labels={"Zaman (sn)": "Zaman (sn)", "Anlık araç": "Araç sayısı"},
        )
        timeline.update_traces(line_color="#1d4e89")
        st.plotly_chart(chart_layout(timeline), use_container_width=True)
    with right:
        st.subheader("Geçen araçlar")
        cumulative = px.line(
            result.frame_stats,
            x="Zaman (sn)",
            y="Toplam geçen",
            labels={"Zaman (sn)": "Zaman (sn)", "Toplam geçen": "Toplam"},
        )
        cumulative.update_traces(line_color="#0f766e")
        st.plotly_chart(chart_layout(cumulative), use_container_width=True)

    st.subheader("Sınıf dağılımı")
    if result.crossings.empty:
        st.info("Sayım çizgisini geçen araç kaydı oluşmadı. Çizgiyi araçların üzerinden geçeceği konuma alın.")
    else:
        counts = result.crossings["Sınıf"].value_counts().rename_axis("Sınıf").reset_index(name="Adet")
        color_map = {class_label(name): f"rgb{color}" for name, color in CLASS_COLORS.items()}
        bars = px.bar(counts, x="Sınıf", y="Adet", color="Sınıf", color_discrete_map=color_map)
        st.plotly_chart(chart_layout(bars), use_container_width=True)
        st.dataframe(result.crossings, use_container_width=True, hide_index=True)
        crossing_csv = result.crossings.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Geçiş kayıtlarını indir",
            data=crossing_csv,
            file_name="crossings.csv",
            mime="text/csv",
        )


def main() -> None:
    st.title("TrafficMind AI")
    st.caption("Eğitilmiş YOLO26n modeli ile araç tespiti, ByteTrack takibi ve sayım çizgisi.")

    metrics = None
    if RESULTS_CSV.is_file():
        metrics = cached_metrics(str(RESULTS_CSV))

    with st.sidebar:
        st.header("Model")
        st.write("`best.pt`")
        if metrics is not None:
            st.write(f"Seçilen epoch: **{metrics['epoch']}**")
            st.write(f"Doğrulama mAP50: **{metrics['map50']:.3f}**")
            st.write(f"Doğrulama mAP50-95: **{metrics['map50_95']:.3f}**")
            st.caption("Değerler eğitim sırasında kaydedilen results.csv dosyasından okunur.")
        else:
            st.warning("Eğitim kaydı bulunamadı.")
        st.divider()
        st.header("Analiz ayarları")
        confidence = st.slider("Güven eşiği", min_value=0.10, max_value=0.80, value=0.25, step=0.05)
        orientation_label = st.radio("Sayım çizgisi", ["Yatay", "Dikey"], horizontal=True)
        line_ratio = st.slider("Çizgi konumu", min_value=0.05, max_value=0.95, value=0.62, step=0.01)
        count_once = st.toggle("Aynı aracı aynı yönde bir kez say", value=True)
        orientation = "horizontal" if orientation_label == "Yatay" else "vertical"

    uploaded = st.file_uploader("Trafik videosu yükleyin", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("MP4, AVI veya MOV formatında bir trafik videosu seçin. Sayım çizgisini yerleştirdikten sonra analizi başlatın.")
        return

    source_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("source_key") != source_key:
        try:
            saved = save_upload(uploaded)
            readable = ensure_decodable_video(saved)
            frame = read_first_frame(readable)
        except Exception as error:
            st.error(f"Video okunamadı: {error}")
            return
        st.session_state["source_key"] = source_key
        st.session_state["source_path"] = str(readable)
        st.session_state["first_frame"] = frame
        st.session_state.pop("analysis", None)

    frame = st.session_state.get("first_frame")
    if frame is None:
        st.error("Videonun ilk karesi okunamadı.")
        return

    st.subheader("Sayım çizgisi")
    st.image(preview_image(frame, orientation, line_ratio), use_container_width=True)
    st.caption("Sarı çizgi, toplam geçen araç sayısının ölçüldüğü sanal çizgidir. Anlık sayı o karede görünen araçları gösterir.")

    if st.button("Analizi başlat", type="primary"):
        progress = st.progress(0.0, text="Video hazırlanıyor")

        def on_progress(done: int, total: int) -> None:
            if total > 0:
                progress.progress(min(done / total, 1.0), text=f"{done} / {total} kare")
            else:
                progress.progress(0.0, text=f"{done} kare işlendi")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_ROOT / "analyses" / stamp / "trafficmind.mp4"
        try:
            result = analyze_video(
                model=cached_model(),
                source_path=Path(st.session_state["source_path"]),
                output_path=output_path,
                confidence=confidence,
                orientation=orientation,
                line_ratio=line_ratio,
                count_once_per_direction=count_once,
                progress_callback=on_progress,
            )
        except Exception as error:
            progress.empty()
            st.error(str(error))
            return
        progress.progress(1.0, text="Analiz tamamlandı")
        st.session_state["analysis"] = result

    analysis = st.session_state.get("analysis")
    if analysis is not None:
        render_results(analysis)


if __name__ == "__main__":
    main()
