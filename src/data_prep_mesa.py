import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- 1. ตั้งค่า Path สำหรับโฟลเดอร์ MESA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == 'src':
    BASE_DIR = os.path.dirname(BASE_DIR)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "MESA-Web_Job_03242664908")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_mesa_profile():
    # --- 2. ค้นหาไฟล์ profile.data ในโฟลเดอร์ ---
    profile_files = glob.glob(os.path.join(RAW_DIR, "**", "profile*.data"), recursive=True)

    if not profile_files:
        print("❌ ไม่พบไฟล์ profile.data ในโฟลเดอร์ โปรดตรวจสอบการแตกไฟล์")
        return

    target_file = profile_files[-1]
    print(f"กำลังเปิดไฟล์ MESA: {os.path.basename(target_file)}")

    # --- 3. อ่านไฟล์ MESA Profile ---
    try:
        df = pd.read_csv(target_file, sep=r'\s+', skiprows=5)

        if 'radius' not in df.columns:
            print("❌ ไม่พบคอลัมน์ radius")
            return
        r_actual = df['radius'].values

        # ตรวจสอบและดึงค่าความดัน (Pressure)
        if 'pressure' in df.columns:
            p_actual = df['pressure'].values
        elif 'logP' in df.columns:
            p_actual = 10 ** df['logP'].values
        elif 'log_P' in df.columns:
            p_actual = 10 ** df['log_P'].values
        else:
            print("❌ ไม่พบคอลัมน์สำหรับความดัน (pressure/logP/log_P)")
            return

        # ตรวจสอบและดึงค่าความหนาแน่น (Density)
        if 'logRho' in df.columns:
            rho_actual = 10 ** df['logRho'].values
        elif 'log_rho' in df.columns:
            rho_actual = 10 ** df['log_rho'].values
        elif 'density' in df.columns:
            rho_actual = df['density'].values
        else:
            print("❌ ไม่พบคอลัมน์สำหรับความหนาแน่น (density/logRho/log_rho)")
            return

        # กลับด้านข้อมูลจาก แก่นกลาง (0) -> ผิวดาว
        r_reversed = r_actual[::-1]
        p_reversed = p_actual[::-1]
        rho_reversed = rho_actual[::-1]

        # Normalize รัศมี
        r_normalized = r_reversed / r_reversed.max()

        # --- 4. กระบวนการทำ 500-Point Interpolation ---
        print("กำลังทำ 500-Point Interpolation ตามเปเปอร์...")

        # 🌟 แก้ไขตรงนี้: เพิ่ม fill_value="extrapolate" เพื่อให้เดาค่าที่ R=0 ได้
        interp_p = interp1d(r_normalized, p_reversed, kind='cubic', fill_value="extrapolate")
        interp_rho = interp1d(r_normalized, rho_reversed, kind='cubic', fill_value="extrapolate")

        r_500 = np.linspace(0.00, 1.00, 500)
        p_500 = interp_p(r_500)
        rho_500 = interp_rho(r_500)

        p_500 = np.clip(p_500, a_min=1e4, a_max=None)
        rho_500 = np.clip(rho_500, a_min=1e-9, a_max=None)

        # --- 5. บันทึกผลลัพธ์ ---
        df_final = pd.DataFrame({
            'Normalized_Radius': r_500,
            'Density': rho_500,
            'Pressure': p_500
        })

        save_path = os.path.join(PROCESSED_DIR, "mesa_profile_500_points.csv")
        df_final.to_csv(save_path, index=False)
        print(f"✅ สำเร็จ! เตรียมข้อมูล 500 ชั้นเรียบร้อยที่: {save_path}")

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")


if __name__ == "__main__":
    process_mesa_profile()