import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# CONFIG

st.set_page_config(
    page_title="SPK Mobil Bekas AHP",
    layout="wide"
)

st.title("Sistem Pendukung Keputusan Pemilihan Mobil Bekas")
st.subheader("Metode AHP (Analytical Hierarchy Process)")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("used_cars.csv")
    return df

df = load_data()

# =====================================================
# PREPROCESSING
# =====================================================

kolom = [
    'brand',
    'model',
    'model_year',
    'milage',
    'engine',
    'price'
]

df = df[kolom]

# Hapus data kosong
df = df.dropna()

# =====================================================
# CLEANING DATA
# =====================================================

# Bersihkan milage
df['milage'] = (
    df['milage']
    .astype(str)
    .str.replace(' mi.', '', regex=False)
    .str.replace(',', '', regex=False)
)

df['milage'] = pd.to_numeric(
    df['milage'],
    errors='coerce'
)

# Bersihkan engine
df['engine'] = (
    df['engine']
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')[0]
)

df['engine'] = pd.to_numeric(
    df['engine'],
    errors='coerce'
)

# Bersihkan price
df['price'] = (
    df['price']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
)

df['price'] = pd.to_numeric(
    df['price'],
    errors='coerce'
)

# Hapus null setelah cleaning
df = df.dropna()

# Gabungkan nama mobil
df['name'] = df['brand'] + " " + df['model']

df = df.head(500)

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Dataset",
        "Perhitungan AHP",
        "Hasil Ranking",
        "Profil Kelompok"
    ]
)

# =====================================================
# HALAMAN DATASET
# =====================================================

if menu == "Dataset":

    st.header("Dataset Mobil Bekas")

    st.write("Jumlah Data :", df.shape[0])
    st.write("Jumlah Kolom :", df.shape[1])

    st.dataframe(df)

# =====================================================
# HALAMAN AHP
# =====================================================

elif menu == "Perhitungan AHP":

    st.header("Perhitungan AHP")

    st.subheader("Input Perbandingan Antar Kriteria")

    # =====================================================
    # INPUT BOBOT
    # =====================================================

    p_vs_y = st.slider(
        "Harga vs Tahun Mobil",
        1, 9, 3
    )

    p_vs_m = st.slider(
        "Harga vs Milage",
        1, 9, 5
    )

    p_vs_e = st.slider(
        "Harga vs Engine",
        1, 9, 4
    )

    y_vs_m = st.slider(
        "Tahun Mobil vs Milage",
        1, 9, 3
    )

    y_vs_e = st.slider(
        "Tahun Mobil vs Engine",
        1, 9, 2
    )

    m_vs_e = st.slider(
        "Milage vs Engine",
        1, 9, 2
    )

    # =====================================================
    # TOMBOL HITUNG
    # =====================================================

    if st.button("Hitung AHP"):

        kriteria = [
            "price",
            "model_year",
            "milage",
            "engine"
        ]

        n = 4

        # =====================================================
        # MATRIX PAIRWISE
        # =====================================================

        matrix = np.array([
            [1, p_vs_y, p_vs_m, p_vs_e],
            [1/p_vs_y, 1, y_vs_m, y_vs_e],
            [1/p_vs_m, 1/y_vs_m, 1, m_vs_e],
            [1/p_vs_e, 1/y_vs_e, 1/m_vs_e, 1]
        ])

        matrix_df = pd.DataFrame(
            matrix,
            columns=kriteria,
            index=kriteria
        )

        st.subheader("Matriks Pairwise Comparison")

        st.dataframe(matrix_df)

        # =====================================================
        # NORMALISASI MATRIX
        # =====================================================

        col_sum = matrix.sum(axis=0)

        normalized_matrix = matrix / col_sum

        normalized_df = pd.DataFrame(
            normalized_matrix,
            columns=kriteria,
            index=kriteria
        )

        st.subheader("Normalisasi Matriks")

        st.dataframe(normalized_df)

        # =====================================================
        # PRIORITY VECTOR
        # =====================================================

        priority_vector = normalized_matrix.mean(axis=1)

        bobot_df = pd.DataFrame({
            "Kriteria": kriteria,
            "Bobot": priority_vector
        })

        st.subheader("Bobot Prioritas")

        st.dataframe(bobot_df)

        # =====================================================
        # CONSISTENCY RATIO
        # =====================================================

        weighted_sum = np.dot(
            matrix,
            priority_vector
        )

        lambda_max = np.mean(
            weighted_sum / priority_vector
        )

        CI = (lambda_max - n) / (n - 1)

        RI_dict = {
            1: 0.00,
            2: 0.00,
            3: 0.58,
            4: 0.90,
            5: 1.12
        }

        RI = RI_dict[n]

        CR = CI / RI

        st.subheader("Consistency Ratio")

        st.metric(
            "Nilai CR",
            round(CR, 4)
        )

        if CR < 0.1:
            st.success("Konsisten")
        else:
            st.error("Tidak Konsisten")

        # =====================================================
        # NORMALISASI DATA
        # =====================================================

        data = df.copy()

        # BENEFIT
        data['year_norm'] = (
            data['model_year'] /
            data['model_year'].max()
        )

        data['engine_norm'] = (
            data['engine'] /
            data['engine'].max()
        )

        # COST
        data['price_norm'] = (
            data['price'].min() /
            data['price']
        )

        data['milage_norm'] = (
            data['milage'].min() /
            data['milage']
        )

        # =====================================================
        # HITUNG SCORE
        # =====================================================

        data['score'] = (
            data['price_norm'] * priority_vector[0] +
            data['year_norm'] * priority_vector[1] +
            data['milage_norm'] * priority_vector[2] +
            data['engine_norm'] * priority_vector[3]
        )

        # =====================================================
        # RANKING
        # =====================================================

        ranking = data[[
            'name',
            'price',
            'model_year',
            'milage',
            'engine',
            'score'
        ]]

        ranking = ranking.sort_values(
            by='score',
            ascending=False
        )

        ranking['Peringkat'] = range(
            1,
            len(ranking) + 1
        )

        st.session_state['ranking'] = ranking

        st.success("Perhitungan AHP Berhasil")

# =====================================================
# HALAMAN RANKING
# =====================================================

elif menu == "Hasil Ranking":

    st.header("Hasil Ranking Mobil Bekas")

    if 'ranking' in st.session_state:

        ranking = st.session_state['ranking']

        st.dataframe(ranking)

        # =====================================================
        # TOP 10
        # =====================================================

        st.subheader("Top 10 Mobil Terbaik")

        top10 = ranking.head(10)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(
            top10['name'],
            top10['score']
        )

        plt.xticks(rotation=90)

        st.pyplot(fig)

        # DOWNLOAD CSV

        csv = ranking.to_csv(index=False)

        st.download_button(
            label="Download Hasil Ranking",
            data=csv,
            file_name="hasil_ranking.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "Silakan hitung AHP terlebih dahulu"
        )

# HALAMAN PROFIL
elif menu == "Profil Kelompok":

    st.header("Profil Kelompok")

    st.write("""
    Nama Kelompok:
    
    1. Anggota 1
    2. Anggota 2
    3. Dimas Proboningrat
    
    Judul:
    Sistem Pendukung Keputusan Pemilihan Mobil Bekas
    Menggunakan Metode AHP Berbasis Streamlit
    """)