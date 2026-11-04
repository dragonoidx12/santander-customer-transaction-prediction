
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="Santander Transaction Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        ROOT / "santander_final_model.joblib"
    )

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

demo = pd.read_csv(ROOT / "santander_demo.csv")

st.title("📊 Santander Customer Transaction Prediction")
st.write("Optimized Gaussian Naive Bayes")

c1, c2, c3 = st.columns(3)
c1.metric("Validation ROC-AUC", "0.888100")
c2.metric("Validation PR-AUC", "0.575299")
c3.metric("Numerical Features", "200")

st.info(
    "Model, 200 anonim sayısal özellik kullanmaktadır. "
    "Gerçek bankacılık sistemlerine bağlı değildir."
)

st.warning(
    "Yalnızca eğitim amaçlıdır. Gerçek müşterilerin "
    "kişisel veya finansal bilgilerini yüklemeyin. "
    "Skorları gerçek bankacılık kararlarında kullanmayın."
)

st.divider()

tab1, tab2 = st.tabs([
    "Örnek profil",
    "CSV ile toplu tahmin"
])

with tab1:
    st.subheader("Yapay müşteri profili")

    st.caption(
        "200 özelliğin medyanlarından üretilmiş "
        "yapay bir profil kullanıyoruz."
    )

    selected_feature = st.selectbox(
        "Değiştirmek istediğin özellik",
        features
    )

    initial_value = float(
        demo[selected_feature].iloc[0]
    )

    new_value = st.number_input(
        "Özellik değeri",
        value=initial_value,
        format="%.4f"
    )

    if st.button(
        "Örnek Tahmin Yap",
        type="primary"
    ):
        X = demo[features].copy()
        X.loc[0, selected_feature] = new_value

        score = float(
            model.predict_proba(X)[0, 1]
        )

        st.metric(
            "Modelin işlem skoru",
            f"{score:.4f}"
        )

        st.caption(
            "Bu, modelin sınıf 1 için ürettiği skordur. "
            "Kalibre edilmiş gerçek olasılık değildir."
        )

    st.download_button(
        "Örnek CSV indir",
        data=demo.to_csv(index=False),
        file_name="santander_demo.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Toplu tahmin")

    uploaded = st.file_uploader(
        "200 özellik içeren CSV dosyasını yükle",
        type=["csv"]
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)

        st.write("Satır sayısı:", len(df))
        st.dataframe(df.head())

        if st.button(
            "CSV Tahminlerini Hesapla",
            type="primary"
        ):
            missing = [
                c for c in features
                if c not in df.columns
            ]

            if missing:
                st.error(
                    "Eksik özellikler: "
                    + ", ".join(missing)
                )

            elif len(df) == 0 or len(df) > 10000:
                st.error(
                    "1 ile 10.000 arasında satır yükle."
                )

            else:
                X = df[features].apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                X = X.replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                if X.isna().any().any():
                    st.error(
                        "Bazı özellikler eksik veya "
                        "sayısal olmayan değerler içeriyor."
                    )
                else:
                    scores = model.predict_proba(X)[:, 1]

                    result = pd.DataFrame({
                        "row_number": np.arange(1, len(X) + 1),
                        "transaction_score": scores
                    })

                    st.success("Tahminler tamamlandı.")
                    st.dataframe(result.head(30))

                    st.download_button(
                        "Sonuçları CSV indir",
                        data=result.to_csv(index=False),
                        file_name="santander_predictions.csv",
                        mime="text/csv"
                    )

st.caption(
    "Kaggle – Santander Customer Transaction Prediction"
)
