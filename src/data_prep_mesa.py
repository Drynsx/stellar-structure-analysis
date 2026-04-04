import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- 1. ตั้งค่า Path สำหรับโฟลเดอร์ MESA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# หากไฟล์นี้อยู่ในโฟลเดอร์ src ให้ถอยกลับมา 1 ขั้น
if os.path.basename(BASE_DIR) == 'src':
    BASE_DIR = os.path.dirname(BASE_DIR)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "MESA-Web_Job_03242664908")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_mesa_profile():
    # --- 2. ค้นหาไฟล์ profile.data ในโฟลเดอร์ ---
    # ใช้ glob ค้นหาไฟล์ที่ขึ้นต้นด้วย profile และลงท้ายด้วย .data
    profile_files = glob.glob(os.path.join(RAW_DIR, "**", "profile*.data"), recursive=True)

    if not profile_files:
        print("❌ ไม่พบไฟล์ profile.data ในโฟลเดอร์ โปรดตรวจสอบการแตกไฟล์")
        return

    # เลือกไฟล์ profile ล่าสุด (มักจะเป็นจุดจบการทำงานของ MESA)
    target_file = profile_files[-1]
    print(f"กำลังเปิดไฟล์ MESA: {os.path.basename(target_file)}")

    # --- 3. อ่านไฟล์ MESA Profile ---
    try:
        # MESA มักจะมี Header 5 บรรทัดแรก บรรทัดที่ 6 คือชื่อคอลัมน์
        df = pd.read_csv(target_file, delim_whitespace=True, skiprows=5)

        # ตรวจสอบว่าคอลัมน์มีอะไรบ้าง (เผื่อเวอร์ชัน MESA ต่างกัน)
        required_cols = ['radius', 'logP', 'logRho']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ ไม่พบคอลัมน์ {col} ในไฟล์ MESA")
                return

        # แปลงข้อมูลจาก Log กลับเป็นค่าปกติ (antilog)
        # รัศมีใน MESA เป็นหน่วย R/R_sun อยู่แล้ว
        r_actual = df['radius'].values
        p_actual = 10 ** df['logP'].values
        rho_actual = 10 ** df['logRho'].values

        # MESA จะเรียงข้อมูลจากผิวดาว -> แก่นกลาง เราต้องกลับด้าน (Reverse) ให้เป็น 0 -> 1
        r_reversed = r_actual[::-1]
        p_reversed = p_actual[::-1]
        rho_reversed = rho_actual[::-1]

        # Normalize รัศมีให้เป็น 0 ถึง 1 เป๊ะๆ
        r_normalized = r_reversed / r_reversed.max()

        # --- 4. กระบวนการทำ 500-Point Interpolation ---
        print("กำลังทำ 500-Point Interpolation ตามเปเปอร์...")
        interp_p = interp1d(r_normalized, p_reversed, kind='cubic')
        interp_rho = interp1d(r_normalized, rho_reversed, kind='cubic')

        # สร้างจุด 500 จุด
        r_500 = np.linspace(0.00, 1.00, 500)
        p_500 = interp_p(r_500)
        rho_500 = interp_rho(r_500)

        # ป้องกันค่าติดลบที่ขอบดาว
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
    process_mesa_profile()()