import os
import tarfile
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- 1. กำหนด Path ให้ตรงกับเครื่องของคุณ ---
BASE_DIR = r"C:\Users\Lenovo\PycharmProjects\PythonProject"

# แก้ไข RAW_DIR ให้ชี้ไปที่โฟลเดอร์ data\raw
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# โฟลเดอร์สำหรับเก็บไฟล์ที่แตกแล้ว และไฟล์ผลลัพธ์ (โค้ดจะสร้างให้เอง)
EXTRACT_DIR = os.path.join(BASE_DIR, "data", "extracted")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ตรวจสอบและสร้างโฟลเดอร์เผื่อไว้
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_and_interpolate():
    # 1. เปลี่ยนให้หาไฟล์ .txz หรือ .tar.gz ใน RAW_DIR
    tar_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.txz') or f.endswith('.tar.gz')]

    if not tar_files:
        print(f"❌ ไม่พบไฟล์ .txz หรือ .tar.gz ในโฟลเดอร์ {RAW_DIR}")
        return

    for tar_name in tar_files:
        tar_path = os.path.join(RAW_DIR, tar_name)
        print(f"\nกำลังเปิดไฟล์: {tar_name}")

        # 2. เปิดไฟล์แบบรองรับ .txz (ใช้โหมด 'r:xz' ถ้าเป็น .txz และ 'r:gz' ถ้าเป็น .tar.gz)
        mode = 'r:xz' if tar_name.endswith('.txz') else 'r:gz'

        try:
            with tarfile.open(tar_path, mode) as tar:
                # ขอดูหน่อยว่าข้างในมีไฟล์นามสกุลอะไรบ้าง
                all_members = tar.getnames()
                print(f"👉 ตัวอย่างไฟล์ข้างใน {tar_name}: {all_members[:5]} ...")

                # ดึงเฉพาะไฟล์ .profile
                profile_members = [m for m in tar.getmembers() if m.name.endswith('.profile')]

                if not profile_members:
                    print(f"⚠️ ไม่พบไฟล์ .profile ใน {tar_name} (ไฟล์นี้อาจมีแค่ข้อมูลพื้นผิว ไม่ใช่โครงสร้างภายใน)")
                    print("-" * 50)
                    continue

                print(f"✅ เยี่ยมมาก! พบไฟล์ .profile จำนวน {len(profile_members)} ไฟล์")

                # *** (สำคัญ!) สั่งแตกไฟล์ออกมาก่อน ไม่อย่างนั้น Pandas จะอ่านข้อมูลไม่ได้ ***
                print("กำลังแตกไฟล์ลงในโฟลเดอร์ extracted...")
                tar.extractall(path=EXTRACT_DIR, members=profile_members[:5])

                # --- 3. เริ่มกระบวนการทำ 500-Point Interpolation ---
                for member in profile_members[:5]:  # ทดสอบทำ 5 ไฟล์ก่อน
                    file_path = os.path.join(EXTRACT_DIR, member.name)

                    try:
                        # อ่านไฟล์ Profile (ข้าม Header 5 บรรทัดแรก)
                        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=5)

                        # แปลงกลับจาก log เป็นค่าปกติ
                        R_actual = 10 ** df['logR']
                        P_actual = 10 ** df['logP']
                        Rho_actual = 10 ** df['logRho']

                        # ทำให้รัศมีอยู่ในช่วง 0 ถึง 1 (Normalized Radius: x = r/R_total)
                        R_max = R_actual.max()
                        x_normalized = R_actual / R_max

                        # กลับด้านข้อมูล (จาก 0 -> 1)
                        x_reversed = x_normalized.values[::-1]
                        P_reversed = P_actual.values[::-1]
                        Rho_reversed = Rho_actual.values[::-1]

                        # สร้างจุด 500 จุด
                        x_500 = np.linspace(0, 1, 500)

                        # Interpolation
                        interp_P = interp1d(x_reversed, P_reversed, kind='cubic', fill_value="extrapolate")
                        interp_Rho = interp1d(x_reversed, Rho_reversed, kind='cubic', fill_value="extrapolate")

                        P_500 = interp_P(x_500)
                        Rho_500 = interp_Rho(x_500)

                        # เก็บใส่ DataFrame ใหม่
                        df_500 = pd.DataFrame({
                            'x_normalized': x_500,
                            'Pressure': P_500,
                            'Density': Rho_500
                        })

                        # บันทึกเป็นไฟล์ CSV (ใช้ os.path.basename เพื่อป้องกัน Error ถ้าใน .tar มี Sub-folder)
                        save_name = os.path.basename(member.name).replace('.profile', '_500pts.csv')
                        save_path = os.path.join(PROCESSED_DIR, save_name)
                        df_500.to_csv(save_path, index=False)
                        print(f"✅ บันทึกโมเดล 500 จุดสำเร็จ: {save_name}")

                    except Exception as e:
                        print(f"❌ เกิดข้อผิดพลาดกับไฟล์ {os.path.basename(member.name)}: {e}")

        except Exception as e:
            print(f"❌ ไม่สามารถเปิดไฟล์ {tar_name} ได้: {e}")


if __name__ == "__main__":
    process_and_interpolate()