import logging
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.charts import plot_anomaly_timeline, plot_comparison, plot_metrics_bar, plot_signal

st.set_page_config(page_title="AEGIS Dashboard", page_icon="🛰️", layout="wide")

st.markdown("""<style>.stApp{background:linear-gradient(135deg,#020408 0%,#0a1628 100%);color:#e8f4ff}h1,h2,h3{color:#00d4ff!important;text-shadow:0 0 10px rgba(0,212,255,0.3)}[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1628 0%,#020408 100%);border-right:1px solid rgba(0,212,255,0.2)}.stButton>button{background:linear-gradient(135deg,#00d4ff,#0088cc);color:#020408;border:none;border-radius:6px;font-weight:bold;padding:10px 24px}[data-testid="stMetricValue"]{color:#00ff88;font-size:2rem}</style>""", unsafe_allow_html=True)

def load_csv_data(f):
    df=pd.read_csv(f)
    if "timestamp" in df.columns:df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    if "value" not in df.columns:st.error("❌ CSV 'value' sütunu içermelidir!");st.stop()
    return df

def run_pipeline_safe(df,config,method):
    try:
        from pipeline.orchestrator import run_pipeline
        return run_pipeline(df,config=config,method=method)
    except Exception as e:st.error(f"❌ Pipeline hatası: {e}");return None

def main():
    st.markdown("<h1 style='text-align:center'>🛰️ AEGIS Dashboard</h1>",unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#8bb8d4'>Satellite Telemetry Radiation-Fault Cleaning Pipeline</p>",unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        data_source=st.radio("Veri Kaynağı",["Sentetik Veri","GOES CSV Yükle"])
        method=st.selectbox("Pipeline Metodu",["classic","ml","both"],index=0)
        with st.expander("🔧 Gelişmiş Ayarlar"):
            zscore_threshold=st.slider("Z-Score Eşiği",2.0,5.0,3.5,0.1)
            iqr_multiplier=st.slider("IQR Çarpanı",1.0,3.0,1.5,0.1)
            window_size=st.slider("Pencere Boyutu",20,100,50,5)
        st.markdown("---");st.markdown("**TUA Astro Hackathon 2026**");st.markdown("Ömer & Ahmet")
    
    if "data_loaded" not in st.session_state:st.session_state["data_loaded"]=False
    if "original_data" not in st.session_state:st.session_state["original_data"]=None
    if "pipeline_result" not in st.session_state:st.session_state["pipeline_result"]=None
    
    if data_source=="Sentetik Veri":
        col1,col2=st.columns([3,1])
        with col1:st.info("ℹ️ Sentetik veri kullanılacak (SEU, TID, gaps, noise)")
        with col2:
            if st.button("🔄 Veri Oluştur",use_container_width=True):
                with st.spinner("Sentetik veri oluşturuluyor..."):
                    try:
                        from data.synthetic_generator import generate_corrupted_dataset
                        _,corrupted_df,_=generate_corrupted_dataset(n=5000,seed=42)
                        st.session_state["original_data"]=corrupted_df
                        st.session_state["data_loaded"]=True
                        st.session_state["pipeline_result"]=None
                        st.success("✅ Sentetik veri oluşturuldu!")
                    except Exception as e:st.error(f"❌ Veri oluşturma hatası: {e}")
    else:
        uploaded_file=st.file_uploader("📁 GOES CSV Dosyası Yükle",type=["csv"])
        if uploaded_file is not None:
            try:
                df=load_csv_data(uploaded_file)
                st.session_state["original_data"]=df
                st.session_state["data_loaded"]=True
                st.session_state["pipeline_result"]=None
                st.success(f"✅ {len(df)} satır veri yüklendi!")
            except Exception as e:st.error(f"❌ Dosya yükleme hatası: {e}")
    
    if st.session_state.get("data_loaded",False):
        tab1,tab2,tab3=st.tabs(["📊 Veri & Anomali Tespiti","🔧 Pipeline & Temizleme","📈 Sonuçlar & Metrikler"])
        
        with tab1:
            st.markdown("### 📊 Veri & Anomali Tespiti")
            original_data=st.session_state.get("original_data")
            if original_data is not None:
                col1,col2,col3=st.columns(3)
                with col1:st.metric("Toplam Örnek",len(original_data))
                with col2:st.metric("NaN Değer",original_data["value"].isna().sum())
                with col3:st.metric("Değer Aralığı",f"{original_data['value'].max()-original_data['value'].min():.2f}")
                st.markdown("#### Ham Sinyal")
                fig_raw=plot_signal(original_data,title="Ham Telemetri Sinyali",color="#f59e0b")
                st.plotly_chart(fig_raw,use_container_width=True,key="tab1_raw_signal")
                with st.expander("📋 Veri Önizleme"):st.dataframe(original_data.head(100),use_container_width=True)
        
        with tab2:
            st.markdown("### 🔧 Pipeline & Temizleme")
            if st.button("▶️ Pipeline'ı Çalıştır",use_container_width=True,type="primary"):
                original_data=st.session_state.get("original_data")
                config={"dsp_detector":{"zscore_threshold":zscore_threshold,"iqr_multiplier":iqr_multiplier,"window":window_size}}
                with st.spinner(f"Pipeline çalışıyor (method={method})..."):
                    result=run_pipeline_safe(original_data,config,method)
                    if result is not None:
                        st.session_state["pipeline_result"]=result
                        st.success("✅ Pipeline başarıyla tamamlandı!")
                        st.rerun()
            pipeline_result=st.session_state.get("pipeline_result")
            if pipeline_result is not None:
                st.markdown("#### Pipeline Sonuçları")
                col1,col2=st.columns(2)
                with col1:
                    st.markdown("##### Orijinal Sinyal")
                    original_data=st.session_state.get("original_data")
                    fig_orig=plot_signal(original_data,title="Orijinal",color="#f59e0b")
                    st.plotly_chart(fig_orig,use_container_width=True,key="tab2_original")
                with col2:
                    st.markdown("##### Temizlenmiş Sinyal")
                    cleaned_data=pipeline_result.get("cleaned_data")
                    fig_clean=plot_signal(cleaned_data,title="Temizlenmiş",color="#00ff88")
                    st.plotly_chart(fig_clean,use_container_width=True,key="tab2_cleaned")
                st.markdown("#### Karşılaştırma")
                fig_comp=plot_comparison(original_data,cleaned_data)
                st.plotly_chart(fig_comp,use_container_width=True,key="tab2_comparison")
        
        with tab3:
            st.markdown("### 📈 Sonuçlar & Metrikler")
            pipeline_result=st.session_state.get("pipeline_result")
            if pipeline_result is None:
                st.info("ℹ️ Pipeline'ı çalıştırmak için 'Pipeline & Temizleme' sekmesine gidin.")
            else:
                metrics=pipeline_result.get("metrics",{})
                fault_timeline=pipeline_result.get("fault_timeline",pd.DataFrame())
                col1,col2,col3=st.columns(3)
                with col1:st.metric("🔍 Tespit Edilen Hata",metrics.get("faults_detected",0))
                with col2:st.metric("✅ Düzeltilen Hata",metrics.get("faults_corrected",0))
                with col3:st.metric("⏱️ İşlem Süresi",f"{metrics.get('processing_time',0):.3f}s")
                st.markdown("---")
                st.markdown("#### Pipeline Metrikleri")
                fig_metrics=plot_metrics_bar(metrics)
                st.plotly_chart(fig_metrics,use_container_width=True,key="tab3_metrics")
                st.markdown("#### Anomali Zaman Çizelgesi")
                fig_timeline=plot_anomaly_timeline(fault_timeline)
                st.plotly_chart(fig_timeline,use_container_width=True,key="tab3_timeline")
                if not fault_timeline.empty:
                    with st.expander("📋 Anomali Detayları"):st.dataframe(fault_timeline,use_container_width=True)
                st.markdown("---");st.markdown("#### 💾 Veri İndirme")
                cleaned_data=pipeline_result.get("cleaned_data")
                if cleaned_data is not None:
                    csv=cleaned_data.to_csv(index=False)
                    st.download_button("📥 Temizlenmiş Veriyi İndir (CSV)",data=csv,file_name="cleaned_telemetry.csv",mime="text/csv",use_container_width=True)
    else:
        st.info("👈 Başlamak için sol menüden veri kaynağı seçin ve veri yükleyin.")

if __name__=="__main__":main()
