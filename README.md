# Stellar Structure Modeling Project

โปรเจกต์นี้เป็นการจำลองโครงสร้างภายในของดวงอาทิตย์ (Stellar Structure) โดยอ้างอิงระเบียบวิธีวิจัยจากการทำ 500-Point Interpolation และการคำนวณ Polytropic Index

## 🗺️ Project Roadmap

- [ ] **Phase 1: Data Acquisition**
  - [x] Set up Git repository and project structure.
  - [ ] Write Python script to fetch Standard Solar Model (SSM) data automatically from the Institute for Advanced Study (IAS).
  
- [ ] **Phase 2: Data Preprocessing (Methodology)**
  - [ ] Parse raw `.dat` file to extract Radius, Density, and Pressure.
  - [ ] Perform Cubic Spline Interpolation.
  - [ ] Generate exactly 500 layers of stellar profile data.
  - [ ] Export to `data/processed/solar_profile_500_points.csv`.

- [ ] **Phase 3: Physics Calculation**
  - [ ] Load the 500-point dataset.
  - [ ] Calculate the Polytropic Index (n) across the radius using the relation between Pressure and Density.
  
- [ ] **Phase 4: Lane-Emden Equation (Future Scope)**
  - [ ] Implement numerical integration (e.g., Runge-Kutta) for the Lane-Emden equation.
  - [ ] Compare derived results with the actual Standard Solar Model.

## 📂 Folder Structure
* `generate_profile_professional.py`: Script สำหรับโหลดและทำ Interpolation
* `data/raw/`: เก็บไฟล์ดั้งเดิมจากฐานข้อมูล
* `data/processed/`: เก็บไฟล์ CSV ที่หั่นเป็น 500 ชั้นแล้ว