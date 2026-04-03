import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# URL สำหรับ MIST v1.2 Stellar Tracks (ข้อมูลละเอียดรายชั้น)
BASE_URL = "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/"
SAVE_DIR = "data/raw/mist_tracks"
os.makedirs(SAVE_DIR, exist_ok=True)


def download_for_part1():
    print("กำลังเชื่อมต่อกับเซิร์ฟเวอร์ MIST...")
    # เน้นดึงข้อมูลที่เป็น vvcritical0.0 (ดาวที่หมุนไม่เร็วมาก) เพื่อให้สอดคล้องกับแบบจำลองโพลีโทรปมาตรฐาน
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    # เลือกเฉพาะช่วงโลหะหนัก (Metallicity) ที่ระบุใน Paper (เช่น p0.00 คือดวงอาทิตย์)
    links = [a['href'] for a in soup.find_all('a', href=True) if 'feh_p0.00' in a['href'] and 'vvcrit0.0' in a['href']]

    for link in links:
        file_url = BASE_URL + link
        target_path = os.path.join(SAVE_DIR, link)

        if os.path.exists(target_path):
            continue

        print(f"กำลังดาวน์โหลดข้อมูลสำหรับตอนที่ 1: {link}")
        res = requests.get(file_url, stream=True)
        total = int(res.headers.get('content-length', 0))

        with open(target_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == "__main__":
    download_for_part1()