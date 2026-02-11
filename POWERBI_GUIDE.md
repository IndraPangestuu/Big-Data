# Panduan Power BI Dashboard - Jakarta Traffic Analytics

## Daftar Isi
1. [Instalasi Power BI Desktop](#1-instalasi-power-bi-desktop)
2. [Import Data ke Power BI](#2-import-data-ke-power-bi)
3. [Membuat Visualisasi](#3-membuat-visualisasi)
4. [Menggunakan Fitur Prediksi](#4-menggunakan-fitur-prediksi)
5. [Refresh Data](#5-refresh-data)

---

## 1. Instalasi Power BI Desktop

### Langkah-langkah:

1. **Download Power BI Desktop**
   - Buka: https://powerbi.microsoft.com/desktop/
   - Klik tombol **"Download Free"**
   - Pilih bahasa (English/Indonesia)
   - Klik **"Next"** lalu **"Download"**

2. **Install Power BI Desktop**
   - Buka file installer yang sudah didownload
   - Tunggu proses install selesai
   - Buka aplikasi Power BI Desktop
   - Sign in dengan akun Microsoft (gratis untuk membuat)

3. **Setting Python (Optional - untuk prediksi)**
   - Di Power BI Desktop, klik **File** > **Options and settings** > **Options**
   - Pilih **Python scripting**
   - Set deteksi Python otomatis atau pilih path manual
   - Klik **OK**

---

## 2. Import Data ke Power BI

### File Data yang Tersedia:

| File | Deskripsi | Baris |
|------|-----------|-------|
| `powerbi_bus_passengers.csv` | Data penumpang & bus per kuartal | 65 |
| `powerbi_traffic.csv` | Data traffic per jam | 105,264 |
| `powerbi_haltes.csv` | Data halte/bus stop | 269 |
| `powerbi_routes.csv` | Data rute/trayek bus | 120 |

### Cara Import:

#### Method A: Load One by One

1. Di Power BI Desktop, klik **Home** > **Get Data** > **Text/CSV**
2. Cari dan pilih file `powerbi_bus_passengers.csv`
3. Klik **Open** lalu **Load**
4. Ulangi langkah 2-3 untuk file lainnya:
   - `powerbi_traffic.csv`
   - `powerbi_haltes.csv`
   - `powerbi_routes.csv`

#### Method B: Load Multiple (Folder)

1. Taruh semua file CSV dalam satu folder
2. Klik **Get Data** > **Folder**
3. Pilih folder yang berisi file CSV
4. Pilih **Combine and Load**

---

## 3. Membuat Visualisasi

### 3.1 Membuat Tab (Halaman)

1. Klik **+** di bagian bawah untuk tambah halaman baru
2. Klik kanan pada tab > **Page Information** > rename halaman:
   - **Overview**
   - **Passenger Analytics**
   - **Traffic Patterns**
   - **Halte & Routes**
   - **Prediction**

---

### 3.2 Tab: Overview

#### KPI Cards
1. Klik **Card** visual
2. Drag field `Passenger_Count` dari table `powerbi_bus_passengers`
3. Repeat untuk `Bus_Count` dan `Passengers_Per_Bus`

#### Quarterly Trend Chart
1. Klik **Line Chart** visual
2. Drag `Year_Quarter` ke **Axis**
3. Drag `Passenger_Count` ke **Values**

#### Hourly Traffic Pattern
1. Klik **Area Chart** visual
2. Drag `Hour` dari table `powerbi_traffic` ke **Axis**
3. Drag `Traffic_Volume` ke **Values**
4. Tambah **Shape** > Rectangle untuk highlight jam sibuk (07:00-09:00, 16:00-19:00)

---

### 3.3 Tab: Passenger Analytics

#### Service Type Distribution
1. Klik **Donut Chart** visual
2. Drag `Service_Type` ke **Legend**
3. Drag `Passenger_Count` ke **Values**

#### Service Type Over Time
1. Klik **Stacked Column Chart**
2. Drag `Year_Quarter` ke **Axis**
3. Drag `Service_Type` ke **Legend**
4. Drag `Passenger_Count` ke **Values**

#### Bus Efficiency
1. Klik **Clustered Bar Chart**
2. Drag `Service_Type` ke **Axis**
3. Drag `Passengers_Per_Bus` ke **Values**
4. Sort by value descending

---

### 3.4 Tab: Traffic Patterns

#### LOS Distribution
1. Klik **Column Chart**
2. Drag `LOS` ke **Axis**
3. Drag `Traffic_Volume` ke **Values** (aggregate: Count)

#### Traffic by Day of Week
1. Klik **Bar Chart**
2. Drag `Day_Name` ke **Axis**
3. Drag `Traffic_Volume` ke **Values** (aggregate: Average)

#### Heatmap (Hour x Day)
1. Klik **Matrix** visual
2. Drag `Day_Name` ke **Rows**
3. Drag `Hour` ke **Columns**
4. Drag `Traffic_Volume` ke **Values** (aggregate: Average)
5. Set conditional formatting: Font color or background

---

### 3.5 Tab: Halte & Routes

#### Halte Map
1. Klik **Filled Map** visual
2. Drag `Latitude` ke **Latitude**
3. Drag `Longitude` ke **Longitude**
4. Drag `Region` ke **Legend**
5. Tampilkan detail nama halte di Tooltips

#### Haltes by Region
1. Klik **Bar Chart**
2. Drag `Region` ke **Axis**
3. Drag Halte Name ke **Values** (Count)

---

### 3.6 Tab: Prediction

#### Single Prediction

Buat calculated table untuk input parameter:

1. **What If Parameter** untuk Tanggal:
   - Modeling > New Parameter
   - Name: `SelectedDate`
   - Data type: Date
   - Current value: 2025-03-15

2. **What If Parameter** untuk Jam:
   - Name: `SelectedHour`
   - Data type: Whole number
   - Min: 0, Max: 23, Increment: 1

3. **What If Parameter** untuk Lokasi:
   - Name: `SelectedLocation`
   - Data type: Text
   - List: Sudirman-Thamrin, Harmoni, Semanggi, Kuningan, Tol Dalam Kota

4. **Python Script** untuk prediksi:
   - Klik **Transform Data** > Run Python Script
   - Paste code dari `prediction_script_powerbi.py`

---

## 4. Menggunakan Fitur Prediksi

### Cara Manual (Tanpa Python Integration):

Jika Python integration sulit, gunakan cara ini:

1. Jalankan script Python terpisah:
   ```bash
   python prediction_script_powerbi.py
   ```

2. Import file hasil:
   - `powerbi_prediction_sample.csv` - Prediksi per jam untuk 1 hari
   - `powerbi_location_comparison.csv` - Perbandingan lokasi

3. Buat visualisasi dari file ini

---

## 5. Refresh Data

### Refresh Manual:
1. Klik **Home** > **Refresh**
2. Power BI akan membaca ulang semua file CSV

### Automatic Refresh:
1. Klik **Home** > **Transform Data**
2. Set refresh interval (Pro license needed)

---

## 6. DAX Measures Penting

### Total Passengers
```
Total Passengers = SUM(powerbi_bus_passengers[Passenger_Count])
```

### Year to Date Passengers
```
YTD Passengers = TOTALYTD([Total Passengers], powerbi_bus_passengers[Date])
```

### Passengers per Bus
```
Passengers per Bus = DIVIDE([Total Passengers], SUM(powerbi_bus_passengers[Bus_Count]))
```

### Is Rush Hour
```
Is Rush Hour = IF(powerbi_traffic[Hour] >= 7 && powerbi_traffic[Hour] <= 9) ||
               (powerbi_traffic[Hour] >= 16 && powerbi_traffic[Hour] <= 19), "Yes", "No")
```

### LOS Category (Calculated Column di data, bukan measure)
```
LOS_Category = SWITCH(
    TRUE(),
    powerbi_traffic[Traffic_Volume] < 20, "A - Free Flow",
    powerbi_traffic[Traffic_Volume] < 40, "B - Good",
    powerbi_traffic[Traffic_Volume] < 60, "C - Fair",
    powerbi_traffic[Traffic_Volume] < 75, "D - Congested",
    powerbi_traffic[Traffic_Volume] < 90, "E - Very Congested",
    "F - Gridlock"
)
```

---

## 7. Tips & Tricks

### Formatting:
- Gunakan conditional formatting untuk highlight nilai tinggi/rendah
- Set custom colors sesuai tema Jakarta
- Add tooltips untuk informasi tambahan

### Performance:
- Hapus kolom yang tidak dipakai
- Use "Import" bukan "DirectQuery" untuk data CSV
- Limit data rows jika terlalu besar

### Export:
- Klik **File** > **Export** > **Analyze in Excel**
- Atau Publish ke Power BI Service untuk sharing online

---

## 8. Troubleshooting

### Error saat buka CSV:
- Pastikan format CSV comma-delimited
- Cek encoding (UTF-8 vs ANSI)

### Python script tidak jalan:
- Pastikan Python terinstall di komputer
- Cek path Python di Power BI Options
- Install library: `pip install pandas numpy scikit-learn joblib`

### Map tidak muncul:
- Cek format Latitude/Longitude harus decimal degrees
- Pastikan internet connect untuk map tiles

---

## 9. File Reference

| File | Kegunaan |
|------|-----------|
| `prepare_powerbi_data.py` | Script untuk generate CSV siap import |
| `prediction_script_powerbi.py` | Script prediksi untuk Python integration |
| `powerbi_*.csv` | File hasil data preparation |
| `traffic_model.pkl` | Model ML untuk prediksi |

---

## 10. Support

Untuk pertanyaan lebih lanjut:
- Power BI Documentation: https://docs.microsoft.com/power-bi/
- Python in Power BI: https://docs.microsoft.com/power-bi/desktop/python-scripting
